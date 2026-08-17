from dataclasses import dataclass, field
import asyncio
import time
import uuid
from collections.abc import AsyncIterator
from .backends import Backend
from .cache import KVCache, CachePressure
from . import metrics
from .log import get_logger


logger = get_logger("scheduler")


@dataclass(order=True)
class Job:
    priority: int
    created: float = field(compare=True)
    prompt: str = field(compare=False)
    max_tokens: int = field(compare=False)
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex, compare=False)
    output: asyncio.Queue = field(default_factory=asyncio.Queue, compare=False)
    cancelled: bool = field(default=False, compare=False)


class Scheduler:
    def __init__(self, backend: Backend, cache: KVCache, max_batch_size=8, window_ms=8, max_queue_size=256):
        self.backend, self.cache = backend, cache
        self.max_batch_size, self.window_ms, self.max_queue_size = max_batch_size, window_ms, max_queue_size
        self.queue: asyncio.PriorityQueue[Job] = asyncio.PriorityQueue(maxsize=max_queue_size)
        self._task: asyncio.Task | None = None
        self.running = False

    async def start(self) -> None:
        if not self.running:
            self.running = True
            self._task = asyncio.create_task(self._loop())
            logger.debug("scheduler started", extra={"queue_size": self.queue.qsize()})

    async def stop(self) -> None:
        self.running = False
        if self._task:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            logger.debug("scheduler stopped")

    async def submit(self, prompt: str, max_tokens: int, priority: int = 0) -> AsyncIterator[str]:
        if self.queue.full():
            logger.warning("request rejected: scheduler queue is full", extra={"queue_size": self.queue.qsize()})
            raise RuntimeError("scheduler admission limit reached")
        job = Job(priority, time.monotonic(), prompt, max_tokens)
        await self.queue.put(job)
        async def stream():
            try:
                while True:
                    item = await job.output.get()
                    if item is None:
                        break
                    yield item
            finally:
                job.cancelled = True
        return stream()

    async def _loop(self) -> None:
        while self.running:
            first = await self.queue.get()
            batch = [first]
            deadline = time.monotonic() + self.window_ms / 1000
            while len(batch) < self.max_batch_size and time.monotonic() < deadline:
                try:
                    batch.append(self.queue.get_nowait())
                except asyncio.QueueEmpty:
                    await asyncio.sleep(0.001)
            metrics.BATCH_SIZE.observe(len(batch))
            await asyncio.gather(*(self._run(job) for job in batch))

    async def _run(self, job: Job) -> None:
        start = time.monotonic()
        try:
            self.cache.allocate(job.request_id, job.max_tokens)
            metrics.QUEUE_WAIT.observe(start - job.created)
            first = True
            count = 0
            async for token in self.backend.generate(job.prompt, job.max_tokens):
                if job.cancelled:
                    break
                if first:
                    metrics.TTFT.observe(time.monotonic() - start)
                    first = False
                count += 1
                metrics.TOKENS.inc()
                await job.output.put(token)
        except CachePressure as exc:
            logger.warning("request failed: cache pressure", extra={"request_id": job.request_id}, exc_info=exc)
            await job.output.put(RuntimeError(str(exc)))
        except Exception as exc:
            logger.exception("request failed in backend", extra={"request_id": job.request_id})
            await job.output.put(exc)
        finally:
            metrics.CACHE_UTILIZATION.set(self.cache.utilization)
            metrics.CACHE_FRAGMENTATION.set(self.cache.fragmentation)
            metrics.LATENCY.observe(time.monotonic() - start)
            self.cache.release(job.request_id)
            await job.output.put(None)

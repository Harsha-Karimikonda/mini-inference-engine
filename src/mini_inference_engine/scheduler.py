import asyncio
import collections
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from . import metrics
from .backends import Backend
from .cache import CachePressure, KVCache
from .errors import AdmissionError
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
        self._active_count: int = 0
        self._latencies: collections.deque[float] = collections.deque(maxlen=32)

    @property
    def queue_depth(self) -> int:
        return self.queue.qsize()

    @property
    def active_count(self) -> int:
        return self._active_count

    @property
    def avg_latency(self) -> float:
        return sum(self._latencies) / len(self._latencies) if self._latencies else 0.0

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
            raise AdmissionError("scheduler admission limit reached")
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
            if hasattr(self.backend, "generate_batch"):
                await self._run_batch(batch)
            else:
                await asyncio.gather(*(self._run(job) for job in batch))

    async def _run_batch(self, batch: list[Job]) -> None:
        start = time.monotonic()
        admitted: list[Job] = []
        for job in batch:
            try:
                self.cache.allocate(job.request_id, job.max_tokens)
                self._active_count += 1
                metrics.QUEUE_WAIT.observe(start - job.created)
                admitted.append(job)
            except CachePressure as exc:
                logger.warning("request failed: cache pressure", extra={"request_id": job.request_id}, exc_info=exc)
                await job.output.put(RuntimeError(str(exc)))
                await job.output.put(None)

        if not admitted:
            return

        metrics.CACHE_UTILIZATION.set(self.cache.utilization)
        metrics.CACHE_FRAGMENTATION.set(self.cache.fragmentation)

        max_tokens = max(j.max_tokens for j in admitted)
        prompts = [j.prompt for j in admitted]
        queues: list[asyncio.Queue[str | None]] = [asyncio.Queue() for _ in admitted]

        async def forward_stream(idx: int, job: Job):
            first = True
            try:
                while True:
                    token = await queues[idx].get()
                    if token is None or job.cancelled:
                        break
                    if first:
                        metrics.TTFT.observe(time.monotonic() - start)
                        first = False
                    metrics.record_tokens()
                    await job.output.put(token)
            except Exception as exc:
                logger.exception("error streaming job output", extra={"request_id": job.request_id})
                await job.output.put(exc)
            finally:
                self._active_count = max(0, self._active_count - 1)
                duration = time.monotonic() - start
                self._latencies.append(duration)
                metrics.LATENCY.observe(duration)
                self.cache.release(job.request_id)
                metrics.CACHE_UTILIZATION.set(self.cache.utilization)
                metrics.CACHE_FRAGMENTATION.set(self.cache.fragmentation)
                await job.output.put(None)

        forwarders = [asyncio.create_task(forward_stream(i, j)) for i, j in enumerate(admitted)]
        try:
            await self.backend.generate_batch(prompts, max_tokens, queues)
        except Exception:
            logger.exception("batch generation failed in backend")
            for q in queues:
                await q.put(None)
        await asyncio.gather(*forwarders)

    async def _run(self, job: Job) -> None:
        self._active_count += 1
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
                metrics.record_tokens()
                await job.output.put(token)
        except CachePressure as exc:
            logger.warning("request failed: cache pressure", extra={"request_id": job.request_id}, exc_info=exc)
            await job.output.put(RuntimeError(str(exc)))
        except Exception as exc:
            logger.exception("request failed in backend", extra={"request_id": job.request_id})
            await job.output.put(exc)
        finally:
            self._active_count = max(0, self._active_count - 1)
            duration = time.monotonic() - start
            self._latencies.append(duration)
            metrics.CACHE_UTILIZATION.set(self.cache.utilization)
            metrics.CACHE_FRAGMENTATION.set(self.cache.fragmentation)
            metrics.LATENCY.observe(duration)
            self.cache.release(job.request_id)
            await job.output.put(None)

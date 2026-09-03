import asyncio
import re
import time
from collections.abc import Callable

import psutil

from .backends import Backend
from .cache import KVCache
from .config import Settings
from .log import get_logger
from .router import Router, Worker
from .scheduler import Scheduler

logger = get_logger("autoscaler")


def estimate_model_memory_gb(model_name: str, quantization: str = "none") -> float:
    """Estimate model weights VRAM footprint in GB based on name and precision."""
    if model_name == "mock":
        return 0.1

    # Extract parameter count from name (e.g. 0.5B, 1.5B, 3B, 7B, 8B, 14B, 70B)
    match = re.search(r"(\d+(?:\.\d+)?)[Bb]", model_name)
    param_billions = float(match.group(1)) if match else 3.0

    # Bytes per parameter based on quantization/precision
    quant_lower = (quantization or "none").lower()
    if quant_lower == "4bit":
        bytes_per_param = 0.65  # 4 bits + quant scale/zero-point overhead
    elif quant_lower == "8bit":
        bytes_per_param = 1.15
    elif quant_lower in ("fp16", "bf16", "float16", "bfloat16"):
        bytes_per_param = 2.0
    elif quant_lower == "fp32":
        bytes_per_param = 4.0
    else:
        bytes_per_param = 2.0

    return round(param_billions * bytes_per_param, 2)


def get_usable_memory_gb(device: str = "auto") -> tuple[float, float]:
    """Returns (total_memory_gb, usable_memory_gb) for the target hardware."""
    try:
        import torch

        if device == "cuda" or (device == "auto" and torch.cuda.is_available()):
            total_bytes = torch.cuda.get_device_properties(0).total_memory
            total_gb = total_bytes / (1024**3)
            usable_gb = max(2.0, total_gb - 2.0)  # Reserve 2 GB for CUDA runtime / OS
            return round(total_gb, 1), round(usable_gb, 1)
    except (ImportError, RuntimeError, AttributeError) as exc:
        logger.debug("CUDA memory query unavailable", exc_info=exc)

    # System RAM (Apple Silicon unified memory or CPU)
    total_gb = psutil.virtual_memory().total / (1024**3)
    usable_gb = max(2.0, total_gb - 4.0)  # Reserve 4 GB for macOS & display
    return round(total_gb, 1), round(usable_gb, 1)


def compute_safe_worker_bounds(
    model_name: str,
    device: str = "auto",
    quantization: str = "none",
    cache_blocks: int = 1024,
    cache_block_tokens: int = 16,
    user_min: int = 1,
    user_max: int = 0,
) -> tuple[int, int, float, float]:
    """Compute safe (min_workers, max_workers, model_memory_gb, usable_memory_gb)."""
    _, usable_gb = get_usable_memory_gb(device)
    model_cost_gb = estimate_model_memory_gb(model_name, quantization)

    # Approximate KV-cache memory pool per worker in GB
    cache_gb = (cache_blocks * cache_block_tokens * 128 * 2) / (1024**3)
    worker_cost_gb = max(0.2, model_cost_gb + cache_gb + 0.3)

    if user_max > 0:
        max_w = user_max
    else:
        hardware_max = int(usable_gb // worker_cost_gb)
        # Device contention cap: On a single GPU or Apple Silicon Metal,
        # 4 workers is the optimal upper bound to avoid command buffer thrashing
        contention_cap = 4 if device in ("mps", "auto") else 8
        max_w = max(1, min(hardware_max, contention_cap))

    min_w = max(1, min(user_min, max_w))
    return min_w, max_w, model_cost_gb, usable_gb


class AutoScaler:
    """Dynamic autoscaler managing the inference worker lifecycle and pool size."""

    def __init__(
        self,
        router: Router,
        settings: Settings,
        backend_factory: Callable[[], Backend],
        eval_interval_s: float = 1.5,
    ):
        self.router = router
        self.settings = settings
        self.backend_factory = backend_factory
        self.eval_interval_s = eval_interval_s
        self.running = False
        self._task: asyncio.Task | None = None
        self.state = "stable"
        self._last_scale_time = time.monotonic()
        self._idle_since: float | None = None
        self._worker_counter = len(router.workers)

        self.min_workers, self.max_workers, self.model_memory_gb, self.usable_memory_gb = compute_safe_worker_bounds(
            settings.model,
            settings.device,
            settings.quantization,
            settings.cache_blocks,
            settings.cache_block_tokens,
            settings.min_workers,
            settings.max_workers,
        )

        logger.info(
            "autoscaler initialized",
            extra={
                "min_workers": self.min_workers,
                "max_workers": self.max_workers,
                "model_memory_gb": self.model_memory_gb,
                "usable_memory_gb": self.usable_memory_gb,
            },
        )

    async def start(self) -> None:
        """Start the background monitoring loop."""
        self.running = True
        self._task = asyncio.create_task(self._loop(), name="autoscaler-loop")
        logger.info("autoscaler background loop started")

    async def stop(self) -> None:
        """Stop the background monitoring loop and cancel pending evaluations."""
        self.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("autoscaler stopped")

    async def _loop(self) -> None:
        while self.running:
            try:
                await asyncio.sleep(self.eval_interval_s)
                await self.evaluate()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.exception("error in autoscaler evaluation loop", exc_info=exc)

    async def evaluate(self) -> None:
        """Evaluate current cluster metrics and trigger scale up / scale down."""
        await self._check_draining_workers()

        self.router.refresh_health()
        active_workers = [w for w in self.router.workers if w.healthy and not w.draining]
        total_workers = len(active_workers)
        total_backlog = sum(w.queue_depth for w in active_workers)
        total_active = sum(w.active for w in active_workers)

        now = time.monotonic()
        cooldown_s = 4.0

        # Scale Up Condition:
        # Total backlog exceeds threshold and pool hasn't hit hardware maximum
        scale_up_condition = (
            (total_backlog >= self.settings.scale_up_threshold * max(1, total_workers) or total_backlog >= 4)
            and total_workers < self.max_workers
            and (now - self._last_scale_time > cooldown_s)
        )

        if scale_up_condition:
            self._idle_since = None
            self.state = "scaling_up"
            logger.info(
                "scaling up worker pool",
                extra={
                    "current_workers": total_workers,
                    "target_workers": total_workers + 1,
                    "backlog": total_backlog,
                    "max_workers": self.max_workers,
                },
            )
            await self.scale_up()
            self._last_scale_time = time.monotonic()
            self.state = "stable"
            return

        # Scale Down Condition:
        # Zero backlog and active load is at or below baseline
        if total_backlog == 0 and total_active <= max(1, total_workers - 1):
            if self._idle_since is None:
                self._idle_since = now
            elif (
                now - self._idle_since >= self.settings.scale_down_idle_s
                and total_workers > self.min_workers
                and now - self._last_scale_time > cooldown_s
            ):
                self.state = "scaling_down"
                logger.info(
                    "scaling down worker pool (idle threshold reached)",
                    extra={
                        "current_workers": total_workers,
                        "target_workers": total_workers - 1,
                        "idle_s": round(now - self._idle_since, 1),
                        "min_workers": self.min_workers,
                    },
                )
                await self.scale_down()
                self._last_scale_time = time.monotonic()
                self._idle_since = None
                self.state = "stable"
        else:
            self._idle_since = None
            self.state = "stable"

    async def scale_up(self) -> Worker:
        """Spawn and register a new worker into the router pool."""
        self._worker_counter += 1
        worker_id = f"worker-{self._worker_counter}"
        existing_ids = {w.worker_id for w in self.router.workers}
        while worker_id in existing_ids:
            self._worker_counter += 1
            worker_id = f"worker-{self._worker_counter}"

        backend = self.backend_factory()
        cache = KVCache(self.settings.cache_blocks, self.settings.cache_block_tokens)
        scheduler = Scheduler(
            backend,
            cache,
            self.settings.max_batch_size,
            self.settings.batch_window_ms,
            self.settings.max_queue_size,
        )
        await scheduler.start()
        worker = Worker(worker_id, scheduler)
        self.router.add_worker(worker)
        logger.info("spawned new autoscaled worker", extra={"worker": worker_id})
        return worker

    async def scale_down(self) -> Worker | None:
        """Mark the least-utilized worker for graceful drain and removal."""
        active_workers = [w for w in self.router.workers if not w.draining]
        if len(active_workers) <= self.min_workers:
            return None

        # Pick candidate with lowest active connections and queue
        candidate = min(active_workers, key=lambda w: (w.active, w.queue_depth))
        self.router.mark_draining(candidate.worker_id)
        logger.info("draining worker for scale-down", extra={"worker": candidate.worker_id})
        return candidate

    async def _check_draining_workers(self) -> None:
        """Check all draining workers and terminate any that have completed their queue."""
        draining = [w for w in self.router.workers if w.draining]
        for worker in draining:
            if worker.queue_depth == 0 and worker.active == 0:
                await self._finalize_drain(worker)

    async def _finalize_drain(self, worker: Worker) -> None:
        """Stop worker scheduler and remove from router."""
        try:
            if hasattr(worker.scheduler, "stop"):
                await worker.scheduler.stop()
        except Exception as exc:
            logger.warning("error stopping scheduler during drain", extra={"worker": worker.worker_id}, exc_info=exc)
        self.router.remove_worker(worker.worker_id)
        logger.info("worker drain completed and terminated", extra={"worker": worker.worker_id})

    def get_status(self) -> dict:
        """Get live autoscaler state and capacity telemetry."""
        active = [w for w in self.router.workers if w.healthy and not w.draining]
        draining = [w for w in self.router.workers if w.draining]
        return {
            "enabled": self.settings.autoscale_enabled,
            "state": self.state,
            "min_workers": self.min_workers,
            "max_workers": self.max_workers,
            "active_workers": len(active),
            "draining_workers": len(draining),
            "hardware_usable_ram_gb": self.usable_memory_gb,
            "estimated_model_memory_gb": self.model_memory_gb,
            "scale_up_threshold": self.settings.scale_up_threshold,
            "scale_down_idle_s": self.settings.scale_down_idle_s,
        }

import time
from dataclasses import dataclass, field

from . import metrics
from .errors import NoHealthyWorkers
from .log import get_logger

logger = get_logger("router")


@dataclass
class Worker:
    worker_id: str
    scheduler: object
    healthy: bool = True
    active: int = 0
    queue_depth: int = 0
    latency: float = 0.05
    draining: bool = False
    last_heartbeat: float = field(default_factory=time.monotonic)


class Router:
    def __init__(self, workers: list[Worker], policy: str = "latency_aware", heartbeat_timeout_s: float = 5):
        self.workers, self.policy, self.heartbeat_timeout_s = workers, policy, heartbeat_timeout_s
        self._cursor = 0

    def add_worker(self, worker: Worker) -> None:
        if any(w.worker_id == worker.worker_id for w in self.workers):
            logger.warning("worker already registered", extra={"worker": worker.worker_id})
            return
        self.workers.append(worker)
        logger.info("worker added to router", extra={"worker": worker.worker_id, "total": len(self.workers)})

    def mark_draining(self, worker_id: str) -> Worker | None:
        for worker in self.workers:
            if worker.worker_id == worker_id:
                worker.draining = True
                logger.info("worker marked for draining", extra={"worker": worker_id})
                return worker
        return None

    def remove_worker(self, worker_id: str) -> Worker | None:
        for i, worker in enumerate(self.workers):
            if worker.worker_id == worker_id:
                removed = self.workers.pop(i)
                logger.info("worker removed from router", extra={"worker": worker_id, "remaining": len(self.workers)})
                return removed
        return None

    def refresh_health(self) -> None:
        now = time.monotonic()
        for worker in self.workers:
            # Local workers do not have a separate heartbeat RPC. Their
            # scheduler's running state is the liveness signal until workers
            # move to the process boundary described by protocol.proto.
            if getattr(worker.scheduler, "running", False):
                worker.last_heartbeat = now
            worker.healthy = now - worker.last_heartbeat <= self.heartbeat_timeout_s
            if hasattr(worker.scheduler, "queue_depth"):
                worker.queue_depth = worker.scheduler.queue_depth
            if hasattr(worker.scheduler, "active_count"):
                worker.active = worker.scheduler.active_count
            if hasattr(worker.scheduler, "avg_latency"):
                worker.latency = worker.scheduler.avg_latency
            metrics.WORKER_HEALTH.labels(worker.worker_id).set(1 if (worker.healthy and not worker.draining) else 0)
            metrics.WORKER_LOAD.labels(worker.worker_id).set(worker.active)

    def choose(self) -> Worker:
        self.refresh_health()
        healthy = [worker for worker in self.workers if worker.healthy and not worker.draining]
        if not healthy:
            logger.error("no healthy workers available")
            raise NoHealthyWorkers("no healthy workers")
        if self.policy == "round_robin":
            worker = healthy[self._cursor % len(healthy)]
            self._cursor += 1
        elif self.policy == "least_connections":
            worker = min(healthy, key=lambda worker: worker.active)
        elif self.policy == "least_queue_length":
            worker = min(healthy, key=lambda worker: worker.queue_depth)
        else:
            def latency_score(w: Worker) -> float:
                lat = w.latency if w.latency > 0 else 0.05
                return (w.queue_depth + w.active + 1) * lat

            worker = min(healthy, key=latency_score)
        logger.debug("worker selected", extra={"worker": worker.worker_id})
        return worker

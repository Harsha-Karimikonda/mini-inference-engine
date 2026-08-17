from dataclasses import dataclass, field
import time
from . import metrics
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
    last_heartbeat: float = field(default_factory=time.monotonic)


class Router:
    def __init__(self, workers: list[Worker], policy: str = "latency_aware", heartbeat_timeout_s: float = 5):
        self.workers, self.policy, self.heartbeat_timeout_s = workers, policy, heartbeat_timeout_s
        self._cursor = 0

    def refresh_health(self) -> None:
        now = time.monotonic()
        for worker in self.workers:
            # Local workers do not have a separate heartbeat RPC. Their
            # scheduler's running state is the liveness signal until workers
            # move to the process boundary described by protocol.proto.
            if getattr(worker.scheduler, "running", False):
                worker.last_heartbeat = now
            worker.healthy = now - worker.last_heartbeat <= self.heartbeat_timeout_s
            metrics.WORKER_HEALTH.labels(worker.worker_id).set(1 if worker.healthy else 0)
            metrics.WORKER_LOAD.labels(worker.worker_id).set(worker.active)

    def choose(self) -> Worker:
        self.refresh_health()
        healthy = [worker for worker in self.workers if worker.healthy]
        if not healthy:
            logger.error("no healthy workers available")
            raise RuntimeError("no healthy workers")
        if self.policy == "round_robin":
            worker = healthy[self._cursor % len(healthy)]
            self._cursor += 1
        elif self.policy == "least_connections":
            worker = min(healthy, key=lambda worker: worker.active)
        elif self.policy == "least_queue_length":
            worker = min(healthy, key=lambda worker: worker.queue_depth)
        else:
            worker = min(healthy, key=lambda worker: (worker.latency, worker.queue_depth, worker.active))
        logger.debug("worker selected", extra={"worker": worker.worker_id})
        return worker

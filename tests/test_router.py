import time

import pytest

from mini_inference_engine.errors import NoHealthyWorkers
from mini_inference_engine.router import Router, Worker


class FakeScheduler: pass

class RunningScheduler:
    running = True

class TelemetryScheduler:
    running = True
    queue_depth = 7
    active_count = 3
    avg_latency = 0.12

def workers():
    return [Worker("a", FakeScheduler(), active=3, queue_depth=4, latency=.4), Worker("b", FakeScheduler(), active=1, queue_depth=2, latency=.2)]

@pytest.mark.parametrize("policy, expected", [("least_connections", "b"), ("least_queue_length", "b"), ("latency_aware", "b")])
def test_policies(policy, expected):
    assert Router(workers(), policy).choose().worker_id == expected

def test_round_robin_and_health():
    pool = workers()
    router = Router(pool, "round_robin", heartbeat_timeout_s=.01)
    pool[0].last_heartbeat = time.monotonic() - 1
    assert router.choose().worker_id == "b"

def test_running_local_worker_refreshes_health():
    worker = Worker("local", RunningScheduler())
    worker.last_heartbeat = time.monotonic() - 1
    router = Router([worker], heartbeat_timeout_s=.01)
    assert router.choose().worker_id == "local"

def test_router_raises_no_healthy_workers():
    pool = workers()
    router = Router(pool, "round_robin", heartbeat_timeout_s=.01)
    now = time.monotonic()
    for w in pool:
        w.last_heartbeat = now - 1
    with pytest.raises(NoHealthyWorkers):
        router.choose()

def test_refresh_health_pulls_scheduler_telemetry():
    worker = Worker("telemetry-worker", TelemetryScheduler(), active=0, queue_depth=0, latency=0.0)
    router = Router([worker])
    router.refresh_health()
    assert worker.queue_depth == 7
    assert worker.active == 3
    assert worker.latency == 0.12


def test_latency_aware_balances_backlog_against_latency():
    # Worker A is slightly faster (0.1s) but heavily backlogged (queue=50)
    # Worker B is slightly slower (0.15s) but completely free (queue=0)
    # Router must choose Worker B because (0+0+1)*0.15 = 0.15s < (50+0+1)*0.1 = 5.1s
    fast_busy = Worker("fast-busy", RunningScheduler(), active=0, queue_depth=50, latency=0.10)
    slow_idle = Worker("slow-idle", RunningScheduler(), active=0, queue_depth=0, latency=0.15)
    router = Router([fast_busy, slow_idle], policy="latency_aware")
    chosen = router.choose()
    assert chosen.worker_id == "slow-idle"


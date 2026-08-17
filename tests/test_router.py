import time
import pytest
from mini_inference_engine.router import Router, Worker

class FakeScheduler: pass

class RunningScheduler:
    running = True

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

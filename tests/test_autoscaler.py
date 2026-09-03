import pytest

from mini_inference_engine.autoscaler import (
    AutoScaler,
    compute_safe_worker_bounds,
    estimate_model_memory_gb,
    get_usable_memory_gb,
)
from mini_inference_engine.backends import MockBackend
from mini_inference_engine.cache import KVCache
from mini_inference_engine.config import Settings
from mini_inference_engine.router import Router, Worker
from mini_inference_engine.scheduler import Scheduler


def test_model_memory_estimation():
    assert estimate_model_memory_gb("mock") == 0.1
    # 0.5B in FP16 = 0.5 * 2.0 = 1.0 GB
    assert estimate_model_memory_gb("Qwen/Qwen2.5-0.5B-Instruct", "none") == 1.0
    # 3B in 4-bit = 3.0 * 0.65 = 1.95 GB
    assert estimate_model_memory_gb("Qwen/Qwen2.5-3B-Instruct", "4bit") == 1.95
    # 7B in 4-bit = 7.0 * 0.65 = 4.55 GB
    assert estimate_model_memory_gb("Qwen/Qwen2.5-7B-Instruct", "4bit") == 4.55


def test_usable_memory_and_safe_bounds():
    total_gb, usable_gb = get_usable_memory_gb("auto")
    assert total_gb > 0
    assert usable_gb > 0

    min_w, max_w, model_cost, usable = compute_safe_worker_bounds(
        "Qwen/Qwen2.5-3B-Instruct",
        device="auto",
        quantization="4bit",
        user_min=1,
        user_max=0,
    )
    assert min_w == 1
    assert max_w >= 1
    assert model_cost > 0
    assert usable > 0


@pytest.mark.asyncio
async def test_autoscaler_scale_up_under_backlog():
    settings = Settings(model="mock", min_workers=1, max_workers=4, scale_up_threshold=2)
    backend_factory = lambda: MockBackend()
    sched1 = Scheduler(backend_factory(), KVCache(64, 16), 4, 10, 64)
    await sched1.start()

    worker1 = Worker("worker-1", sched1)
    router = Router([worker1])

    autoscaler = AutoScaler(router, settings, backend_factory, eval_interval_s=0.1)
    autoscaler._last_scale_time = 0.0  # bypass cooldown for test

    assert len(router.workers) == 1

    class MockSched:
        queue_depth = 5
        active_count = 0
        avg_latency = 0.05
        running = True

    worker1.scheduler = MockSched()
    await autoscaler.evaluate()

    # Worker pool should have scaled up
    assert len(router.workers) == 2
    assert router.workers[1].worker_id == "worker-2"

    await sched1.stop()
    await router.workers[1].scheduler.stop()


@pytest.mark.asyncio
async def test_autoscaler_scale_down_and_drain():
    settings = Settings(model="mock", min_workers=1, max_workers=4, scale_down_idle_s=0.1)
    backend_factory = lambda: MockBackend()

    sched1 = Scheduler(backend_factory(), KVCache(64, 16), 4, 10, 64)
    sched2 = Scheduler(backend_factory(), KVCache(64, 16), 4, 10, 64)
    await sched1.start()
    await sched2.start()

    worker1 = Worker("worker-1", sched1)
    worker2 = Worker("worker-2", sched2)
    router = Router([worker1, worker2])

    autoscaler = AutoScaler(router, settings, backend_factory, eval_interval_s=0.05)
    autoscaler._last_scale_time = 0.0

    # Both idle: trigger scale down
    autoscaler._idle_since = 0.0  # simulate idle for long enough
    await autoscaler.evaluate()

    # One worker should be marked draining
    draining = [w for w in router.workers if w.draining]
    assert len(draining) == 1

    # Finalize drain
    await autoscaler._finalize_drain(draining[0])
    assert len(router.workers) == 1
    assert not router.workers[0].draining

    await sched1.stop()

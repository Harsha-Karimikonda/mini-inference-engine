import pytest

from mini_inference_engine.backends import MockBackend
from mini_inference_engine.cache import KVCache
from mini_inference_engine.errors import AdmissionError
from mini_inference_engine.scheduler import Scheduler


@pytest.mark.asyncio
async def test_scheduler_streams_tokens():
    scheduler = Scheduler(MockBackend(delay=0), KVCache(64, 4), max_batch_size=2)
    await scheduler.start()
    stream = await scheduler.submit("hello", 3)
    tokens = [token async for token in stream]
    await scheduler.stop()
    assert len(tokens) == 3

@pytest.mark.asyncio
async def test_scheduler_admission_error_on_full_queue():
    # Don't start scheduler, so items remain queued
    scheduler = Scheduler(MockBackend(delay=0), KVCache(64, 4), max_queue_size=1)
    await scheduler.submit("first", 2)
    assert scheduler.queue_depth == 1
    with pytest.raises(AdmissionError):
        await scheduler.submit("second", 2)

@pytest.mark.asyncio
async def test_scheduler_telemetry_properties():
    scheduler = Scheduler(MockBackend(delay=0.01), KVCache(64, 4), max_batch_size=2)
    assert scheduler.queue_depth == 0
    assert scheduler.active_count == 0
    assert scheduler.avg_latency == 0.0

    await scheduler.start()
    stream = await scheduler.submit("hello", 2)
    tokens = [token async for token in stream]
    assert len(tokens) == 2
    await scheduler.stop()

    assert scheduler.active_count == 0
    assert scheduler.avg_latency > 0.0

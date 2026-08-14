import pytest
from mini_inference_engine.backends import MockBackend
from mini_inference_engine.cache import KVCache
from mini_inference_engine.scheduler import Scheduler

@pytest.mark.asyncio
async def test_scheduler_streams_tokens():
    scheduler = Scheduler(MockBackend(delay=0), KVCache(64, 4), max_batch_size=2)
    await scheduler.start()
    stream = await scheduler.submit("hello", 3)
    tokens = [token async for token in stream]
    await scheduler.stop()
    assert len(tokens) == 3

import pytest
from mini_inference_engine.cache import KVCache, CachePressure

def test_allocate_release_and_reuse():
    cache = KVCache(4, 4)
    first = cache.allocate("a", 5)
    assert len(first.blocks) == 2
    cache.release("a")
    assert cache.utilization == 0
    second = cache.allocate("b", 16)
    assert len(second.blocks) == 4

def test_pressure_is_explicit():
    cache = KVCache(2, 4)
    cache.allocate("a", 8)
    with pytest.raises(CachePressure):
        cache.allocate("b", 1)

def test_fragmentation_is_bounded():
    cache = KVCache(8, 1)
    cache.allocate("a", 2)
    cache.allocate("b", 2)
    cache.release("a")
    assert 0 <= cache.fragmentation <= 1

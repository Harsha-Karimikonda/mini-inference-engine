from mini_inference_engine.prefix_cache import PrefixCache


class DummyLayer:
    def __init__(self, seq_len: int = 16):
        self.keys = type("Tensor", (), {
            "shape": [1, 4, seq_len, 64],
            "clone": lambda self: self,
            "__getitem__": lambda self, idx: self
        })()
        self.values = type("Tensor", (), {
            "shape": [1, 4, seq_len, 64],
            "clone": lambda self: self,
            "__getitem__": lambda self, idx: self
        })()


class DummyDynamicCache:
    def __init__(self, seq_len: int = 16):
        self.layers = [DummyLayer(seq_len)]

    def get_seq_length(self):
        return self.layers[0].keys.shape[2]


def test_chunk_hash_computation():
    cache = PrefixCache(block_tokens=16)
    # 35 tokens = 2 blocks of 16 + 3 remainder
    token_ids = list(range(100, 135))
    chunks = cache._compute_chunk_hashes(token_ids)
    assert len(chunks) == 2
    assert chunks[0][1] == 16
    assert chunks[1][1] == 32
    assert isinstance(chunks[0][0], str)
    assert chunks[0][0] != chunks[1][0]


def test_prefix_cache_hit_and_stats():
    cache = PrefixCache(block_tokens=16, max_cached_blocks=64)

    # Prefix of 32 tokens
    shared_prefix = list(range(10, 42))
    query1 = shared_prefix + [99, 100, 101]
    query2 = shared_prefix + [201, 202, 203]

    # Query 1: Miss and Insert
    pkv1, matched = cache.match(query1)
    assert pkv1 is None
    assert matched == 0
    assert cache.misses == 1

    dummy_pkv = DummyDynamicCache(seq_len=32)
    cache.insert(query1, dummy_pkv)

    stats = cache.stats()
    assert stats["cached_entries"] == 2
    assert stats["cached_blocks"] == 2

    # Query 2: Hit!
    _pkv2, matched2 = cache.match(query2)
    # Note: clone_dynamic_cache checks isinstance(cache, DynamicCache),
    # so for pure DummyDynamicCache it returns None or we can verify match logic
    assert matched2 == 32
    assert cache.hits == 1
    assert cache.tokens_saved == 32

    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["hit_rate_pct"] == 50.0
    assert stats["tokens_saved"] == 32


def test_prefix_cache_lru_eviction():
    # Cache capacity = 4 blocks (e.g. two 32-token prefixes)
    cache = PrefixCache(block_tokens=16, max_cached_blocks=4)

    prefix_a = list(range(100, 132))  # 2 blocks
    prefix_b = list(range(200, 232))  # 2 blocks
    prefix_c = list(range(300, 332))  # 2 blocks

    dummy_pkv = DummyDynamicCache(seq_len=32)

    cache.insert(prefix_a, dummy_pkv)
    cache.insert(prefix_b, dummy_pkv)
    assert cache._total_blocks == 4

    # Inserting prefix_c should evict prefix_a
    cache.insert(prefix_c, dummy_pkv)
    assert cache._total_blocks == 4

    # prefix_a should be evicted
    chunks_a = cache._compute_chunk_hashes(prefix_a)
    assert chunks_a[-1][0] not in cache._entries

    # prefix_c should be present
    chunks_c = cache._compute_chunk_hashes(prefix_c)
    assert chunks_c[-1][0] in cache._entries

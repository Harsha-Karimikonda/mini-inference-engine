import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass

from . import metrics
from .log import get_logger

logger = get_logger("prefix_cache")


def clone_dynamic_cache(cache: object, max_length: int | None = None) -> object | None:
    """Fast clone and slice for Hugging Face DynamicCache without full deepcopy."""
    if cache is None:
        return None
    try:
        from transformers import DynamicCache

        if isinstance(cache, DynamicCache):
            new_cache = DynamicCache()
            for i, layer in enumerate(cache.layers):
                k = layer.keys
                v = layer.values
                if max_length is not None and k.shape[2] > max_length:
                    k = k[:, :, :max_length, :]
                    v = v[:, :, :max_length, :]
                new_cache.update(k.clone(), v.clone(), i)
            return new_cache
        if hasattr(cache, "layers"):
            return cache
    except Exception as exc:
        logger.debug("unable to clone DynamicCache", exc_info=exc)
    return None


@dataclass
class CachedBlock:
    hash_key: str
    token_ids: tuple[int, ...]
    pkv: object
    num_tokens: int
    blocks_count: int
    last_accessed: float


class PrefixCache:
    """Chunk-based cross-request KV prefix cache with LRU eviction."""

    def __init__(self, block_tokens: int = 16, max_cached_blocks: int = 512):
        self.block_tokens = block_tokens
        self.max_cached_blocks = max_cached_blocks
        self._entries: OrderedDict[str, CachedBlock] = OrderedDict()
        self._total_blocks = 0
        self.hits = 0
        self.misses = 0
        self.tokens_saved = 0

    def _compute_chunk_hashes(self, token_ids: list[int]) -> list[tuple[str, int]]:
        """Returns chained hashes for every 16-token block boundary."""
        results = []
        num_blocks = len(token_ids) // self.block_tokens
        curr_hash = ""
        for b in range(num_blocks):
            chunk = tuple(token_ids[b * self.block_tokens : (b + 1) * self.block_tokens])
            h = hashlib.sha256(f"{curr_hash}:{chunk}".encode()).hexdigest()[:16]
            curr_hash = h
            results.append((h, (b + 1) * self.block_tokens))
        return results

    def match(self, token_ids: list[int]) -> tuple[object | None, int]:
        """Find longest matching cached prefix. Returns (cloned_pkv, matched_tokens)."""
        chunks = self._compute_chunk_hashes(token_ids)
        if not chunks:
            self.misses += 1
            metrics.PREFIX_CACHE_MISSES.inc()
            return None, 0

        # Longest match first
        for chunk_hash, num_tokens in reversed(chunks):
            if chunk_hash in self._entries:
                entry = self._entries[chunk_hash]
                self._entries.move_to_end(chunk_hash)
                entry.last_accessed = time.monotonic()
                cloned_pkv = clone_dynamic_cache(entry.pkv, max_length=num_tokens)
                if cloned_pkv is not None:
                    self.hits += 1
                    self.tokens_saved += num_tokens
                    metrics.PREFIX_CACHE_HITS.inc()
                    metrics.PREFIX_CACHE_SAVED_TOKENS.inc(num_tokens)
                    logger.debug(
                        "prefix cache hit",
                        extra={"matched_tokens": num_tokens, "hash": chunk_hash},
                    )
                    return cloned_pkv, num_tokens

        self.misses += 1
        metrics.PREFIX_CACHE_MISSES.inc()
        return None, 0

    def insert(self, token_ids: list[int], pkv: object) -> None:
        """Cache prompt prefix blocks at every 16-token boundary."""
        if pkv is None:
            return
        num_blocks = len(token_ids) // self.block_tokens
        if num_blocks == 0:
            return

        chunks = self._compute_chunk_hashes(token_ids)
        for chunk_hash, num_tokens in chunks:
            if chunk_hash in self._entries:
                self._entries.move_to_end(chunk_hash)
                continue

            cached_pkv = clone_dynamic_cache(pkv, max_length=num_tokens)
            if cached_pkv is None:
                continue

            # Evict oldest entries if capacity exceeded
            while (self._total_blocks + 1) > self.max_cached_blocks and self._entries:
                oldest_hash, oldest_entry = self._entries.popitem(last=False)
                self._total_blocks -= oldest_entry.blocks_count
                logger.debug("evicted prefix cache entry", extra={"hash": oldest_hash})

            entry = CachedBlock(
                hash_key=chunk_hash,
                token_ids=tuple(token_ids[:num_tokens]),
                pkv=cached_pkv,
                num_tokens=num_tokens,
                blocks_count=1,
                last_accessed=time.monotonic(),
            )
            self._entries[chunk_hash] = entry
            self._total_blocks += 1
            logger.debug(
                "cached prefix block",
                extra={"hash": chunk_hash, "tokens": num_tokens},
            )

    def stats(self) -> dict:
        """Get live prefix cache hit/miss statistics."""
        total_queries = self.hits + self.misses
        hit_rate = round((self.hits / total_queries * 100), 1) if total_queries > 0 else 0.0
        return {
            "enabled": True,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate_pct": hit_rate,
            "tokens_saved": self.tokens_saved,
            "cached_entries": len(self._entries),
            "cached_blocks": self._total_blocks,
            "max_cached_blocks": self.max_cached_blocks,
        }

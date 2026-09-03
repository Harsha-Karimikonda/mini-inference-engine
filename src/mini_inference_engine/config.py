import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    model: str = "mock"
    device: str = "auto"
    max_batch_size: int = 8
    batch_window_ms: int = 8
    max_queue_size: int = 256
    max_tokens: int = 128
    cache_blocks: int = 256
    cache_block_tokens: int = 16
    routing_policy: str = "latency_aware"
    heartbeat_timeout_s: float = 5.0

    @classmethod
    def from_env(cls) -> "Settings":
        def integer(name: str, default: int) -> int:
            return int(os.getenv(name, default))

        return cls(
            model=os.getenv("MINI_MODEL", "mock"),
            device=os.getenv("MINI_DEVICE", "auto"),
            max_batch_size=integer("MINI_MAX_BATCH_SIZE", 8),
            batch_window_ms=integer("MINI_BATCH_WINDOW_MS", 8),
            max_queue_size=integer("MINI_MAX_QUEUE_SIZE", 256),
            max_tokens=integer("MINI_MAX_TOKENS", 128),
            cache_blocks=integer("MINI_CACHE_BLOCKS", 256),
            cache_block_tokens=integer("MINI_CACHE_BLOCK_TOKENS", 16),
            routing_policy=os.getenv("MINI_ROUTING_POLICY", "latency_aware"),
            heartbeat_timeout_s=float(os.getenv("MINI_HEARTBEAT_TIMEOUT_S", "5.0")),
        )

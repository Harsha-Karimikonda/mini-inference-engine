import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    model: str = "mock"
    device: str = "auto"
    max_batch_size: int = 8
    batch_window_ms: int = 8
    max_queue_size: int = 256
    max_tokens: int = 2048
    cache_blocks: int = 1024
    cache_block_tokens: int = 16
    routing_policy: str = "latency_aware"
    quantization: str = "none"
    heartbeat_timeout_s: float = 5.0
    autoscale_enabled: bool = True
    min_workers: int = 2
    max_workers: int = 0  # 0 means auto-detect based on hardware & model
    scale_up_threshold: int = 3
    scale_down_idle_s: float = 20.0

    @classmethod
    def from_env(cls) -> "Settings":
        def integer(name: str, default: int) -> int:
            return int(os.getenv(name, default))

        def boolean(name: str, default: bool) -> bool:
            val = os.getenv(name)
            return default if val is None else val.lower() in ("true", "1", "yes")

        return cls(
            model=os.getenv("MINI_MODEL", "mock"),
            device=os.getenv("MINI_DEVICE", "auto"),
            max_batch_size=integer("MINI_MAX_BATCH_SIZE", 8),
            batch_window_ms=integer("MINI_BATCH_WINDOW_MS", 8),
            max_queue_size=integer("MINI_MAX_QUEUE_SIZE", 256),
            max_tokens=integer("MINI_MAX_TOKENS", 2048),
            cache_blocks=integer("MINI_CACHE_BLOCKS", 1024),
            cache_block_tokens=integer("MINI_CACHE_BLOCK_TOKENS", 16),
            routing_policy=os.getenv("MINI_ROUTING_POLICY", "latency_aware"),
            quantization=os.getenv("MINI_QUANTIZATION", "none").lower(),
            heartbeat_timeout_s=float(os.getenv("MINI_HEARTBEAT_TIMEOUT_S", "5.0")),
            autoscale_enabled=boolean("MINI_AUTOSCALE_ENABLED", True),
            min_workers=integer("MINI_MIN_WORKERS", 2),
            max_workers=integer("MINI_MAX_WORKERS", 0),
            scale_up_threshold=integer("MINI_SCALE_UP_THRESHOLD", 3),
            scale_down_idle_s=float(os.getenv("MINI_SCALE_DOWN_IDLE_S", "20.0")),
        )

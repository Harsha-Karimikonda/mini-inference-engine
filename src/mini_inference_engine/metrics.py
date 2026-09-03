import collections
import time

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

REQUESTS = Counter("mini_requests_total", "Requests", ["endpoint", "status"])
TOKENS = Counter("mini_tokens_total", "Generated tokens")
TOKEN_THROUGHPUT = Gauge("mini_tokens_per_second", "Estimated token generation throughput per second")
QUEUE_WAIT = Histogram("mini_queue_wait_seconds", "Time waiting in scheduler")
TTFT = Histogram("mini_ttft_seconds", "Time to first token")
LATENCY = Histogram("mini_request_latency_seconds", "End-to-end latency")
BATCH_SIZE = Histogram("mini_batch_size", "Scheduler batch size")
WORKER_HEALTH = Gauge("mini_worker_health", "Worker health", ["worker"])
WORKER_LOAD = Gauge("mini_worker_active_requests", "Worker active requests", ["worker"])
CACHE_UTILIZATION = Gauge("mini_cache_utilization_ratio", "KV cache utilization")
CACHE_FRAGMENTATION = Gauge("mini_cache_fragmentation_ratio", "KV cache fragmentation")
RETRIES = Counter("mini_retries_total", "Requests retried")

_TOKEN_RECORDS: collections.deque[tuple[float, int]] = collections.deque(maxlen=2000)


def record_tokens(count: int = 1) -> None:
    TOKENS.inc(count)
    _TOKEN_RECORDS.append((time.monotonic(), count))


def get_tokens_per_second(window_s: float = 5.0) -> float:
    now = time.monotonic()
    cutoff = now - window_s
    while _TOKEN_RECORDS and _TOKEN_RECORDS[0][0] < cutoff:
        _TOKEN_RECORDS.popleft()
    if not _TOKEN_RECORDS:
        val = 0.0
    else:
        total = sum(c for t, c in _TOKEN_RECORDS)
        elapsed = max(0.5, now - _TOKEN_RECORDS[0][0])
        val = total / elapsed
    TOKEN_THROUGHPUT.set(val)
    return val


def metrics_response() -> tuple[bytes, str]:
    get_tokens_per_second()
    return generate_latest(), CONTENT_TYPE_LATEST

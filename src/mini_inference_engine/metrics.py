from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST

REQUESTS = Counter("mini_requests_total", "Requests", ["endpoint", "status"])
TOKENS = Counter("mini_tokens_total", "Generated tokens")
QUEUE_WAIT = Histogram("mini_queue_wait_seconds", "Time waiting in scheduler")
TTFT = Histogram("mini_ttft_seconds", "Time to first token")
LATENCY = Histogram("mini_request_latency_seconds", "End-to-end latency")
BATCH_SIZE = Histogram("mini_batch_size", "Scheduler batch size")
WORKER_HEALTH = Gauge("mini_worker_health", "Worker health", ["worker"])
WORKER_LOAD = Gauge("mini_worker_active_requests", "Worker active requests", ["worker"])
CACHE_UTILIZATION = Gauge("mini_cache_utilization_ratio", "KV cache utilization")
CACHE_FRAGMENTATION = Gauge("mini_cache_fragmentation_ratio", "KV cache fragmentation")
RETRIES = Counter("mini_retries_total", "Requests retried")


def metrics_response() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST

# Next Steps: Control-Plane Cleanup → Worker Process Split

The project is a working in-process inference control plane prototype. The next milestone is to make scheduling and routing operate on real telemetry, then split workers into separate processes behind the existing `protocol.proto` boundary.

This plan is sequenced into two phases. **Phase 1** is a set of quick, self-contained improvements that make the existing control plane observably correct. **Phase 2** is the structural change: compiling the proto, implementing the gRPC worker service, and wiring the gateway to talk to out-of-process workers.

---

## Phase 1 — Control-Plane Cleanup (Quick Wins)

These changes are independent of the process split and make the current system materially better. Each can be landed as its own commit.

---

### 1.1 Live Scheduler Telemetry → Router

**Problem**: [`Worker.queue_depth`](file:///Users/hkarimkonda/Documents/mini-inference-engine/src/mini_inference_engine/router.py#L16) is always `0` and [`Worker.latency`](file:///Users/hkarimkonda/Documents/mini-inference-engine/src/mini_inference_engine/router.py#L17) is a fixed `0.05`. The `least_queue_length` and `latency_aware` routing policies make no informed decisions.

**Changes**:

#### [MODIFY] [scheduler.py](file:///Users/hkarimkonda/Documents/mini-inference-engine/src/mini_inference_engine/scheduler.py)
- Add read-only properties `queue_depth → int` (returns `self.queue.qsize()`) and `active_count → int` (track in-flight `_run` calls with an `int` counter incremented/decremented in `_run`).
- Add a rolling `avg_latency → float` property. Track the last N (e.g. 32) request latencies in a `collections.deque(maxlen=32)` and expose the mean. Default to `0.0` when empty.

#### [MODIFY] [router.py](file:///Users/hkarimkonda/Documents/mini-inference-engine/src/mini_inference_engine/router.py)
- In `refresh_health()`, read live telemetry from each worker's scheduler:
  ```python
  worker.queue_depth = worker.scheduler.queue_depth
  worker.active = worker.scheduler.active_count
  worker.latency = worker.scheduler.avg_latency
  ```
- This replaces the manual `worker.active += 1` / `worker.active -= 1` bookkeeping that currently lives in [`api.py`](file:///Users/hkarimkonda/Documents/mini-inference-engine/src/mini_inference_engine/api.py#L136) (lines 136, 166, 171), which can be removed.

#### [MODIFY] [api.py](file:///Users/hkarimkonda/Documents/mini-inference-engine/src/mini_inference_engine/api.py)
- Remove the manual `worker.active += 1` and `worker.active = max(0, worker.active - 1)` statements in the `generate()` function (lines 136, 166, 171). The router now reads this from the scheduler directly.

#### [MODIFY] [test_router.py](file:///Users/hkarimkonda/Documents/mini-inference-engine/tests/test_router.py)
- Update `FakeScheduler` to expose `queue_depth`, `active_count`, and `avg_latency` properties so existing routing-policy tests still pass.
- Add a test that verifies `refresh_health()` actually updates `Worker` fields from the scheduler.

---

### 1.2 Admission Control: Queue-Full → HTTP 429

**Problem**: When the scheduler queue is full, [`scheduler.submit()`](file:///Users/hkarimkonda/Documents/mini-inference-engine/src/mini_inference_engine/scheduler.py#L48) raises a bare `RuntimeError("scheduler admission limit reached")`. This bubbles up as a generic 500 in [`api.py`](file:///Users/hkarimkonda/Documents/mini-inference-engine/src/mini_inference_engine/api.py#L140) because the `try/except` at line 133 only catches `RuntimeError` from `router.choose()`.

**Changes**:

#### [NEW] [errors.py](file:///Users/hkarimkonda/Documents/mini-inference-engine/src/mini_inference_engine/errors.py)
- Define a small exception hierarchy:
  ```python
  class AdmissionError(RuntimeError):
      """Scheduler queue is full."""

  class NoHealthyWorkers(RuntimeError):
      """No workers available to serve requests."""
  ```

#### [MODIFY] [scheduler.py](file:///Users/hkarimkonda/Documents/mini-inference-engine/src/mini_inference_engine/scheduler.py)
- `submit()` raises `AdmissionError` instead of bare `RuntimeError`.

#### [MODIFY] [router.py](file:///Users/hkarimkonda/Documents/mini-inference-engine/src/mini_inference_engine/router.py)
- `choose()` raises `NoHealthyWorkers` instead of bare `RuntimeError`.

#### [MODIFY] [api.py](file:///Users/hkarimkonda/Documents/mini-inference-engine/src/mini_inference_engine/api.py)
- Catch `AdmissionError` from `worker.scheduler.submit()` and return HTTP **429** with `Retry-After` header.
- Catch `NoHealthyWorkers` from `router.choose()` and return HTTP **503**.
- Both return OpenAI-shaped error JSON.

#### [NEW] [test_errors.py](file:///Users/hkarimkonda/Documents/mini-inference-engine/tests/test_errors.py)
- Test that a request hitting a full queue gets a `429` response.
- Test that a request with no healthy workers gets a `503` response.

---

### 1.3 Fix Chat Streaming SSE Object Type

**Problem**: The chat streaming endpoint ([`api.py` line 160](file:///Users/hkarimkonda/Documents/mini-inference-engine/src/mini_inference_engine/api.py#L160)) emits chunks with `"object": "text_completion.chunk"`. OpenAI's specification requires `"object": "chat.completion.chunk"` with a `delta` field instead of `text`.

**Changes**:

#### [MODIFY] [api.py](file:///Users/hkarimkonda/Documents/mini-inference-engine/src/mini_inference_engine/api.py)
- The `generate()` function currently builds SSE payloads inline. Refactor so the caller (`completions` or `chat_completions`) can pass an `endpoint` label **and** a chunk-formatter callable.
- For `/v1/completions`: chunks use `"object": "text_completion.chunk"`, `"choices": [{"text": token, ...}]` (unchanged).
- For `/v1/chat/completions`: chunks use `"object": "chat.completion.chunk"`, `"choices": [{"delta": {"content": token}, ...}]`.

#### [MODIFY] [test_api.py](file:///Users/hkarimkonda/Documents/mini-inference-engine/tests/test_api.py)
- Add a streaming chat test that asserts chunks contain `"chat.completion.chunk"` and `"delta"`.

---

### 1.4 Request IDs and Usage Accounting

**Problem**: Request IDs are generated from a monotonic timestamp ([`api.py` line 137](file:///Users/hkarimkonda/Documents/mini-inference-engine/src/mini_inference_engine/api.py#L137): `"cmpl-" + str(int(started * 1000000))`), which is not guaranteed unique under concurrency. Responses also lack the OpenAI `usage` field.

**Changes**:

#### [MODIFY] [api.py](file:///Users/hkarimkonda/Documents/mini-inference-engine/src/mini_inference_engine/api.py)
- Use `uuid.uuid4().hex` for request IDs (prefix `cmpl-` for completions, `chatcmpl-` for chat).
- Add a `usage` dict to non-streaming responses: `{"prompt_tokens": <len>, "completion_tokens": <count>, "total_tokens": <sum>}`. Token counts can use a simple `len(prompt.split())` heuristic for the mock backend; this is a control-plane accounting concern, not a tokenizer concern.

#### [MODIFY] [test_api.py](file:///Users/hkarimkonda/Documents/mini-inference-engine/tests/test_api.py)
- Assert `usage` is present and has the expected shape in completion and chat responses.

---

### 1.5 Add `/v1/models` Endpoint

**Problem**: OpenAI-compatible clients expect `GET /v1/models` to discover available models.

**Changes**:

#### [MODIFY] [api.py](file:///Users/hkarimkonda/Documents/mini-inference-engine/src/mini_inference_engine/api.py)
- Add a `GET /v1/models` endpoint returning:
  ```json
  {"object": "list", "data": [{"id": "<model>", "object": "model", "owned_by": "mini-together"}]}
  ```

#### [MODIFY] [test_api.py](file:///Users/hkarimkonda/Documents/mini-inference-engine/tests/test_api.py)
- Test that `/v1/models` returns a list containing the configured model.

---

## Phase 2 — Worker RPC Boundary

This is the structural change that moves workers into separate processes communicating over gRPC. After Phase 1, the routing layer is already reading live telemetry and returning proper HTTP status codes, so the process split builds on a correct control plane.

---

### 2.1 Compile Proto & Generate gRPC Stubs

**Changes**:

#### [MODIFY] [pyproject.toml](file:///Users/hkarimkonda/Documents/mini-inference-engine/pyproject.toml)
- Move `grpcio` and `grpcio-tools` from optional `[grpc]` to core `dependencies` (they are now required for the gateway).
- Keep `protobuf` in core dependencies.

#### [MODIFY] [Makefile](file:///Users/hkarimkonda/Documents/mini-inference-engine/Makefile)
- Add a `proto` target:
  ```makefile
  proto:
  	$(PYTHON) -m grpc_tools.protoc -Isrc/mini_inference_engine \
  	  --python_out=src/mini_inference_engine \
  	  --grpc_python_out=src/mini_inference_engine \
  	  src/mini_inference_engine/protocol.proto
  ```

#### [NEW] [protocol_pb2.py](file:///Users/hkarimkonda/Documents/mini-inference-engine/src/mini_inference_engine/protocol_pb2.py)
#### [NEW] [protocol_pb2_grpc.py](file:///Users/hkarimkonda/Documents/mini-inference-engine/src/mini_inference_engine/protocol_pb2_grpc.py)
- Generated files. Added to `.gitignore` or committed — either approach works. I'll commit them so the project works without running `make proto` after clone.

---

### 2.2 Worker gRPC Server

A standalone process that owns a `Scheduler`, `Backend`, and `KVCache`, and exposes them over the proto-defined `Worker` service.

**Changes**:

#### [NEW] [worker_service.py](file:///Users/hkarimkonda/Documents/mini-inference-engine/src/mini_inference_engine/worker_service.py)
- Implements the `WorkerServicer` from the generated stubs:
  - `Register` — stores worker ID and model name, returns `accepted=True`.
  - `Heartbeat` — returns current `active_count`, `queue_depth`, `healthy` from scheduler state.
  - `Generate` — calls `scheduler.submit()` and yields `Token` messages on the stream. Handles `AdmissionError` → error token.
  - `Cancel` — sets `job.cancelled = True` on the matching job.
  - `Drain` — stops accepting new jobs, waits for in-flight to complete, then stops the scheduler.
- Has its own `__main__` block for standalone execution:
  ```bash
  python -m mini_inference_engine.worker_service --port 50051 --worker-id worker-1
  ```

#### [MODIFY] [config.py](file:///Users/hkarimkonda/Documents/mini-inference-engine/src/mini_inference_engine/config.py)
- Add worker-specific settings: `worker_port: int`, `worker_id: str`.

#### [NEW] [test_worker_service.py](file:///Users/hkarimkonda/Documents/mini-inference-engine/tests/test_worker_service.py)
- Unit test that starts the gRPC server in-process, sends a `GenerateRequest`, and collects tokens until `done=True`.
- Test that `Heartbeat` returns the correct queue depth and active count.

---

### 2.3 Gateway gRPC Client & Remote Worker

Replace the current in-process `Worker` dataclass with a `RemoteWorker` that communicates over gRPC.

**Changes**:

#### [NEW] [worker_client.py](file:///Users/hkarimkonda/Documents/mini-inference-engine/src/mini_inference_engine/worker_client.py)
- `RemoteWorker` class wrapping a gRPC channel:
  - Exposes `queue_depth`, `active_count`, `avg_latency` properties by periodically calling the `Heartbeat` RPC (or caching the last heartbeat response).
  - `submit(prompt, max_tokens, priority)` → calls `Generate` RPC and returns an async iterator of tokens.
  - `cancel(request_id)` → calls `Cancel` RPC.
  - `drain()` → calls `Drain` RPC.
- Implements a `scheduler`-like interface so the router can read telemetry from it the same way it reads from a local `Scheduler` (duck typing / protocol class).

#### [MODIFY] [router.py](file:///Users/hkarimkonda/Documents/mini-inference-engine/src/mini_inference_engine/router.py)
- Replace the `scheduler: object` type hint on `Worker` with a `Protocol` class that both `Scheduler` and `RemoteWorker` satisfy:
  ```python
  class WorkerBackend(Protocol):
      queue_depth: int
      active_count: int
      avg_latency: float
      running: bool
      async def submit(self, prompt: str, max_tokens: int, priority: int = 0) -> AsyncIterator[str]: ...
  ```

#### [MODIFY] [api.py](file:///Users/hkarimkonda/Documents/mini-inference-engine/src/mini_inference_engine/api.py)
- In the `lifespan`, read a new config flag (`MINI_WORKERS` env var, e.g. `localhost:50051,localhost:50052`).
  - If set: create `RemoteWorker` instances from the addresses.
  - If not set (default): create local in-process workers as before (preserving the zero-setup Mac demo).

#### [MODIFY] [config.py](file:///Users/hkarimkonda/Documents/mini-inference-engine/src/mini_inference_engine/config.py)
- Add `workers: str = ""` setting (comma-separated `host:port` list, empty means local).

---

### 2.4 Docker Compose Multi-Process Demo

**Changes**:

#### [NEW] [Dockerfile.worker](file:///Users/hkarimkonda/Documents/mini-inference-engine/Dockerfile.worker)
- Worker-specific Dockerfile that runs `python -m mini_inference_engine.worker_service`.

#### [MODIFY] [docker-compose.yml](file:///Users/hkarimkonda/Documents/mini-inference-engine/docker-compose.yml)
- Add two worker services (`worker-1`, `worker-2`) using `Dockerfile.worker`.
- Update gateway to set `MINI_WORKERS=worker-1:50051,worker-2:50051`.
- Workers expose port 50051 internally; gateway connects over the Docker network.

---

### 2.5 Integration Tests

#### [NEW] [tests/test_integration.py](file:///Users/hkarimkonda/Documents/mini-inference-engine/tests/test_integration.py)
- **Worker crash/restart**: Start a worker, kill it, verify the gateway marks it unhealthy and routes to the remaining worker. Restart it, verify it re-registers.
- **Queue saturation**: Submit `max_queue_size + 1` requests and verify the last one returns HTTP 429.
- **Client disconnect cancellation**: Start a streaming request, disconnect the client, verify the job is cancelled (token generation stops).
- **Cache pressure**: Configure a tiny cache, submit requests that exceed it, verify `CachePressure` is returned as a structured error.
- **Routing telemetry**: Submit requests with a `least_queue_length` policy, verify the router distributes load based on actual queue depths.

---

## Verification Plan

### Automated Tests
```bash
make install   # install with [dev] and [grpc] extras
make proto     # generate gRPC stubs
make test      # python -m pytest -q
```
All existing tests must continue to pass. New tests cover:
- Scheduler telemetry properties
- Router reading live telemetry
- HTTP 429/503 error responses
- Chat streaming chunk format
- Request ID uniqueness and `usage` fields
- `/v1/models` endpoint
- gRPC worker service round-trip
- Integration scenarios (queue saturation, disconnect, crash recovery)

### Manual Verification
- `make run` still works with zero config (local in-process workers).
- `docker compose up --build` starts gateway + 2 remote workers + Prometheus + Grafana.
- `make benchmark` against the docker-compose setup shows correct routing distribution in Grafana.

---

## Open Questions

> [!IMPORTANT]
> **Worker discovery**: The plan uses a static `MINI_WORKERS` env var for worker addresses. Should we also support dynamic worker registration (workers call `Register` RPC on the gateway at startup) for a more production-like topology? This would be more complex but avoids hardcoding addresses.

> [!IMPORTANT]
> **Phase scope**: Phase 2 is substantial. Would you prefer to land Phase 1 first as a standalone PR, then tackle Phase 2 separately? Or should we implement everything in one pass on the `dev` branch?

> [!NOTE]
> **Python version**: The project requires Python ≥3.11 but only `/usr/bin/python3` (3.9.6) is available on the system. We'll need to install a newer Python (via Homebrew, pyenv, or similar) before we can run tests. Should I include that setup in the plan?

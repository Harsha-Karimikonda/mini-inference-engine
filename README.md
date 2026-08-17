# Mini-Together

A small, observable inference control plane inspired by production LLM serving systems. The default backend is deterministic and requires no model download; an optional Transformers backend supports CPU, Apple MPS, and CUDA.

## Quickstart on macOS

Use Python 3.11 or newer (the system Python 3.9 is too old):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
make install
make run
```

Try `curl localhost:8000/health`, then POST an OpenAI-shaped request to `/v1/completions` or `/v1/chat/completions`. Add `"stream":true` for OpenAI-style SSE. Run `make test` and `make benchmark` in separate terminals. `/metrics` is Prometheus-compatible.

## Real model backend

Install `pip install -e '.[gpu]'`, then set `MINI_MODEL` to a small Hugging Face causal LM and optionally `MINI_DEVICE=mps` or `cuda`. Tests and local smoke runs should keep `MINI_MODEL=mock` for deterministic behavior.

## Compose observability demo

With Docker Desktop installed, run `docker compose up --build`. The gateway is on port 8000, Prometheus on 9090, and Grafana on 3000. The gateway starts two independently scheduled worker actors; `src/mini_inference_engine/protocol.proto` defines the versioned network-worker boundary for the next deployment phase.

## Design

The gateway validates OpenAI-shaped requests, selects a healthy worker using configurable routing, and returns either JSON or OpenAI-style SSE. Each worker owns a bounded priority/FIFO scheduler, dynamic batching window, backend, and logical paged KV-cache allocator. The mock backend makes throughput and failure tests reproducible; the Transformers backend is a simple reference runtime, leaving optimized kernels and speculative decoding for phase two.

## Logging

The service uses the centralized `mini_inference_engine.log` logger. Logs are
written to stderr at `INFO` level by default. Configure them with:

```bash
MINI_LOG_LEVEL=DEBUG MINI_LOG_FORMAT=json python -m mini_inference_engine.server
```

`MINI_LOG_FORMAT` accepts `text` (the default) or `json`. Request logs include
endpoint, worker, status, request ID, and duration; prompts and generated text
are never logged.

# Architecture

The FastAPI gateway owns request validation and worker selection. Each worker owns a bounded scheduler, backend, and logical paged KV cache. The scheduler collects requests during a short batching window, observes queue wait and TTFT, and streams each backend result independently. This keeps the control-plane interfaces useful even when the underlying backend is replaced by a real batched runtime.

`protocol.proto` is the versioned boundary for moving workers into separate processes. The current default deliberately runs two worker actors in one Python process so the Mac mock demo has no network or GPU setup requirement.

Health is heartbeat-based. Router selection ignores workers whose heartbeat has expired and supports round-robin, least-connections, least-queue-length, and latency-aware policies through `MINI_ROUTING_POLICY`.

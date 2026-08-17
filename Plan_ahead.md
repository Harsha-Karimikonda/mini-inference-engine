The next level is to turn this from a strong **in-process serving prototype** into a **real worker-based inference service**.

Your clearest next milestone:

> Split workers into separate processes using the already-defined `protocol.proto`, then make scheduling and routing operate on real worker telemetry.

Why this is the right move:

- The README and architecture already point to this boundary.
- The current two “workers” share one Python process, so a blocking Transformers generation can still impact the whole gateway.
- Routing is mostly conceptual today: `queue_depth` and worker latency are not updated from scheduler activity, so policies such as `least_queue_length` cannot make informed choices.
- The scheduler batches jobs for concurrent execution, but the Transformers backend still calls `model.generate()` independently per request—so it is not true model-level continuous batching.

I’d sequence it like this:

1. **Worker RPC boundary**  
   Implement a worker service/client around `protocol.proto`; gateway owns validation/routing, workers own model, scheduler, and cache.

2. **Real scheduler telemetry and admission control**  
   Publish queue depth, active sequences, TTFT, token throughput, cache pressure, and capacity. Return a clean `429`/`503` when saturated rather than letting scheduler admission errors become generic server failures.

3. **True batched decoding**  
   Replace per-request `model.generate()` with a decode loop that batches active sequences, supports cancellation, and streams tokens as they arrive.

4. **Production-compatible API details**  
   Add request IDs, usage accounting, model listing, standard OpenAI error fields, and correct chat streaming chunks (the chat endpoint currently streams `text_completion.chunk` payloads).

5. **Reliability tests**  
   Add integration tests for worker failure/restart, queue saturation, disconnect cancellation, cache pressure, routing decisions from real telemetry, and streaming correctness.

A small but high-value near-term cleanup: expose scheduler queue depth to the router and add an exception-to-HTTP mapping for a full queue. That makes the existing control plane observably correct before the process split.

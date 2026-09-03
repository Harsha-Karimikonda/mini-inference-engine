# Plan Ahead: Extreme Edition 🚀

Transforming **Mini-Together (`mini-inference-engine`)** from a local serving prototype into an extreme, production-grade, globally distributed AI inference platform using **Cloudflare Workers ($5/mo Paid)** and **Modal ($30 Credits)**.

---

## 1. The Core Infrastructure & Cost Calculus

| Layer | Provider | Plan / Resource | Capabilities & Limits | Cost / Lifetime |
| :--- | :--- | :--- | :--- | :--- |
| **Edge Control Plane** | Cloudflare Workers | Paid (\$5/mo) | • 10M requests/mo included<br>• Workers KV (1M reads/writes)<br>• D1 SQL (25B row reads/mo)<br>• Queues (1M ops/mo)<br>• Vectorize (30M queries/mo)<br>• Durable Objects (in-memory state)<br>• Workers AI (free daily quota) | **\$5 / month** |
| **GPU Compute Plane** | Modal | Pay-as-you-go (\$30 credit) | • Serverless T4 (\$0.59/hr) → **~50 hours**<br>• Serverless L4 (\$0.80/hr) → **~37 hours**<br>• Serverless A10G (\$1.10/hr) → **~27 hours**<br>• Scale-to-zero ($0 idle)<br>• Modal Volumes & Scheduled Cron | **\$30 initial credit** |

---

## 2. Global Architecture Diagram

```
                       [Client / SDK / Webhook]
                                  │
                                  ▼ (HTTPS / WSS / SSE)
┌────────────────────────────────────────────────────────────────────────┐
│ CLOUDFLARE WORKERS EDGE (Global PoPs, <20ms)                           │
│                                                                        │
│  ┌───────────────────────┐  ┌───────────────────────────────────────┐  │
│  │ AI Firewall / Guard   │  │ Semantic Cache & RAG                  │  │
│  │ • PII Redaction       │  │ • Vectorize + Workers KV              │  │
│  │ • Prompt Injection    │  │ • Top-K edge document context         │  │
│  └───────────────────────┘  └───────────────────────────────────────┘  │
│  ┌───────────────────────┐  ┌───────────────────────────────────────┐  │
│  │ Durable Objects       │  │ Edge Models & Draft                   │  │
│  │ • User Session State  │  │ • Workers AI (free tier models)       │  │
│  │ • Sticky KV Routing   │  │ • Draft tokens for speculation        │  │
│  │ • Token Billing / DO  │  │ • Instant cold-start masking          │  │
│  └───────────────────────┘  └───────────────────────────────────────┘  │
│  ┌───────────────────────┐  ┌───────────────────────────────────────┐  │
│  │ Cloudflare Queues     │  │ Edge Analytics & Logs                 │  │
│  │ • Offline batch jobs  │  │ • D1 SQLite (Token / TTFT metrics)    │  │
│  │ • Webhook dispatcher  │  │ • Replaces Prometheus / Grafana       │  │
│  └───────────────────────┘  └───────────────────────────────────────┘  │
└────────────────────────────────────┬───────────────────────────────────┘
                                     │
                        Secure Subrequests / mTLS
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│ MODAL SERVERLESS GPU WORKERS (T4 / L4 / A10G)                          │
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ mini-inference-engine Worker Runtime                             │  │
│  │ • True Continuous Batching Loop                                  │  │
│  │ • Paged KV-Cache & Dynamic Preemption                            │  │
│  │ • Multi-LoRA Hot-Swapping from Modal Volumes                     │  │
│  │ • Speculative Verification Engine                                │  │
│  │ • Client Disconnect Cooperative Cancellation                     │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ Scheduled Jobs & Background Tasks                                │  │
│  │ • Nightly Auto-Fine-Tuning Loop (Modal Cron @ 2 AM)              │  │
│  │ • High-Throughput Queue Drainer                                  │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. The 18 Extreme Capabilities (5 Tiers)

### Tier 1: Edge Intelligence (Zero GPU Cost)

1. **Speculative Edge Pre-Generation**:
   - Cloudflare Durable Objects track multi-turn conversation trees. While User Turn $N$ is being returned, the edge predicts Turn $N+1$ based on common dialog branches and kicks off background generation on Modal. Perceived TTFT drops to **0ms** on hit.
   - *Codebase hook*: Extends `router.py:choose()` to issue speculative `scheduler.submit()`.

2. **Edge Function Calling & Tool Execution**:
   - When the LLM outputs a tool call (`{"tool": "lookup", "args": {...}}`), Cloudflare intercepts the SSE stream, executes the tool at the edge (HTTP API, D1 query, KV read), injects the output into the context, and continues the stream without client round-trips.
   - *Codebase hook*: Intercepts token stream in `api.py:173` `events()` generator.

3. **A/B Testing & Smart Model Cascade**:
   - 3-tier routing: Trivial queries (classification, formatting) → Cloudflare Workers AI ($0); Medium queries → Modal T4; Complex queries → Modal A10G. D1 logs conversion and quality scores.
   - *Codebase hook*: Outer tier logic wrapping `router.py:Router`.

4. **Real-Time Token Economy & Multi-Tenant Billing**:
   - Durable Objects maintain atomic per-user token balances. As SSE chunks stream, tokens are decremented in real time. If the balance hits zero, the stream is halted immediately with `402 Payment Required`.
   - *Codebase hook*: Connects to `metrics.py:TOKENS` and `api.py:159` usage accounting.

---

### Tier 2: GPU Compute Patterns (Modal Orchestration)

5. **Multi-LoRA Hot-Swap per Request**:
   - Base model weights reside once in GPU VRAM. Hundreds of domain-specific LoRA adapters (medical, legal, coding) are mounted from a persistent Modal Volume. Requests specify `"model": "base:lora-id"`.
   - *Codebase hook*: Refactor `backends.py:TransformersBackend` into a `LoRABackend` using HuggingFace PEFT.

6. **True Continuous Batching + Network-Level Cooperative Cancellation**:
   - Replace sequential `model.generate()` with an iterative token-by-token decode loop. When a user closes the browser or cancels, Cloudflare aborts the connection, calls Modal's `CancelRequest` (from `protocol.proto`), freeing the paged KV-cache slots immediately mid-stream.
   - *Codebase hook*: Implements Phase 3 from `Plan_ahead.md`; wires `protocol.proto:CancelRequest` to `scheduler.py:Job.cancelled`.

7. **Nightly Self-Improving Fine-Tune Loop**:
   - High-quality user interactions logged to Cloudflare R2 trigger a Modal Cron job (`@app.function(schedule=modal.Cron("0 2 * * *"))`). It trains a LoRA adapter for 30 minutes (~$0.30 on T4), validates against benchmark evals, and deploys the new adapter to production automatically.

8. **Edge-Draft / GPU-Verify Speculative Decoding**:
   - Cloudflare Workers AI drafts $K$ candidate tokens at the edge in ~50ms. Modal's primary GPU verifies all $K$ tokens in a single parallel forward pass.
   - *Codebase hook*: Implements a speculative decoding wrapper in `backends.py`.

---

### Tier 3: Data Pipeline & Async Workloads

9. **Full Edge RAG Pipeline (Zero GPU Retrieval)**:
   - User documents uploaded via Cloudflare Worker are stored in R2. Embeddings are indexed into Cloudflare Vectorize. At query time, top-$K$ chunks are retrieved in <10ms and injected into the prompt before hitting Modal.

10. **OpenAI-Style Async Batch API with Webhook Delivery**:
    - Submit 10,000 requests in JSONL format. Cloudflare Workers queues them in Cloudflare Queues with a `202 Accepted`. Modal spins up, saturates continuous batching to 100% capacity, dumps outputs to R2, and fires a webhook to the client callback.

11. **Infinite Conversation Memory**:
    - Durable Objects store conversation sliding windows; older turns are summarized and embedded into Vectorize. Relevant past turns are recalled semantically, giving models infinite memory without paying for huge prompt contexts.

---

### Tier 4: Platform, Observability & Security

12. **Serverless Observability Dashboard (Replacing Grafana + Prometheus)**:
    - Instead of hosting resource-heavy Prometheus and Grafana containers, every request streams its TTFT, latency, and queue wait times directly into Cloudflare D1. A single Worker serves an ultra-lightweight HTML dashboard rendering SQL metrics in real time.
    - *Codebase hook*: Replaces `metrics.py:metrics_response()` with asynchronous D1 telemetry ingestion.

13. **Geo-Aware Multi-Region Routing with Data Residency**:
    - Inspect `request.cf.country`. Route EU queries to Modal EU containers and US queries to Modal US containers for GDPR compliance, with automatic fallback if a region's GPU is cold.

14. **Dual-Layer AI Firewall**:
    - *Input Layer*: Blocks prompt injections, jailbreaks, and PII at the edge before wasting GPU compute.
    - *Output Layer*: Scans streaming tokens on the fly to prevent hallucinated API keys or sensitive data leaks.

---

### Tier 5: Truly Extreme / Experimental

15. **Progressive Model Distillation to Edge**:
    - Continuously distill production traffic from your large Modal model into a quantized edge model running on Cloudflare Workers AI. Over time, 70-80% of common queries shift permanently to edge execution, driving GPU bills to near $0.

16. **Multi-Agent Orchestration Engine (Durable Object DAG Conductor)**:
    - Durable Objects orchestrate multi-agent workflows (e.g. Planner → Coder → Critic). Each sub-agent invokes `mini-inference-engine` with specialized system prompts and LoRAs, handling retries, dependencies, and state persistence.

17. **WebSocket Bidirectional Realtime Protocol**:
    - Durable Objects terminate persistent WebSockets for real-time voice and conversational interruption. Allows clients to interrupt generation mid-sentence, instantly sending a cancellation signal to the Modal worker.

18. **Federated Inference Marketplace**:
    - Transform `mini-inference-engine` into a commercial API platform (like Together AI or OpenRouter). Cloudflare handles user authentication, Stripe payment processing, and rate limits, allowing you to monetize your Modal GPU capacity.

---

## 4. Codebase Architecture Mapping

| Existing Codebase Component | Current Role | Extreme Evolution |
| :--- | :--- | :--- |
| `src/mini_inference_engine/api.py` | FastAPI gateway & SSE streaming | Cloudflare Worker edge proxy with token billing, tool calling, and input firewall |
| `src/mini_inference_engine/router.py` | In-memory worker selection | Edge-to-GPU multi-tier routing (Workers AI vs T4 vs A10G) + Sticky prefix routing |
| `src/mini_inference_engine/scheduler.py` | Priority queue & batching window | Continuous batching decode loop with cooperative aborts from Cloudflare disconnects |
| `src/mini_inference_engine/cache.py` | Logical paged KV-cache | Prefix caching paired with Cloudflare Durable Object affinity routing |
| `src/mini_inference_engine/backends.py` | Mock / Hugging Face Transformers | Dynamic PEFT Multi-LoRA swapping + Speculative decoding verifier |
| `src/mini_inference_engine/protocol.proto` | gRPC service definition | Network boundary implemented via Modal web endpoints & WebSocket bridges |

---

## 5. Execution Roadmap

### Phase 1: Baseline Edge-to-Modal Bridge
- [ ] Create Modal container definition (`modal_worker.py`) running `mini_inference_engine.server` on an Nvidia T4 with `scaledown_window=120`.
- [ ] Deploy Cloudflare Worker proxy (`wrangler`) handling OpenAI-compatible `/v1/chat/completions` with streaming pass-through and API key authentication.
- [ ] Configure basic Workers KV caching for exact prompt matches.

### Phase 2: Observability & True Continuous Batching
- [ ] Implement D1 logging for latency, TTFT, token counts, and queue wait times (retiring Prometheus/Grafana overhead).
- [ ] Refactor `scheduler.py` and `backends.py` into a continuous token-by-token batching loop.
- [ ] Connect client disconnection events to `protocol.proto:CancelRequest`.

### Phase 3: Edge Intelligence & Acceleration
- [ ] Integrate Cloudflare Vectorize for semantic similarity caching (>0.96 cosine similarity cache hit).
- [ ] Configure Cloudflare Workers AI as a Tier-1 fallback and draft model for speculative decoding.
- [ ] Implement sticky routing via Durable Objects to maximize KV-cache reuse on multi-turn chats.

### Phase 4: Autonomous Operations & Monetization
- [ ] Deploy Modal Cron nightly fine-tuning pipeline reading interaction datasets from Cloudflare R2.
- [ ] Implement D1-backed token metering, user quotas, and Stripe webhook top-ups.
- [ ] Deploy the real-time serverless observability dashboard.


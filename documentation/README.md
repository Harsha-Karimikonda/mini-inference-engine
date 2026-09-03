# Mini-Together Documentation

Welcome to the documentation for **Mini-Together (`mini-inference-engine`)**.

This folder consolidates all architectural designs, performance guidelines, and strategic roadmaps for the engine.

---

## Documents Index

| Document | Description |
| :--- | :--- |
| [Architecture](file:///Users/hkarimkonda/Documents/mini-inference-engine/documentation/architecture.md) | Comprehensive system architecture: FastAPI gateway, telemetry-aware routing, dynamic batching, paged KV-cache, autoscaling, hardware backends, and RPC boundaries. |
| [Design Notes](file:///Users/hkarimkonda/Documents/mini-inference-engine/documentation/design.md) | Details on KV-cache memory management, token streaming semantics, retries, and cancellation. |
| [Performance](file:///Users/hkarimkonda/Documents/mini-inference-engine/documentation/performance.md) | Benchmarking guide, latency comparisons (p50/p95), TTFT, and throughput metrics. |
| [Plan Ahead](file:///Users/hkarimkonda/Documents/mini-inference-engine/documentation/Plan_ahead.md) | Core roadmap for splitting workers into separate processes, scheduler telemetry, and continuous batching. |
| [Plan Ahead (Extreme Edition)](file:///Users/hkarimkonda/Documents/mini-inference-engine/documentation/Plan_ahead_extreme.md) | Production architecture pairing Cloudflare Workers edge control plane with Modal serverless GPU compute. |


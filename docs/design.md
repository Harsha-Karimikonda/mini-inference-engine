# Design notes

The KV cache is a logical allocator, not a replacement for a framework's internal attention cache. It makes block allocation, release, reuse, fragmentation, and pressure visible and testable. Eviction is restricted to allocations explicitly marked idle.

Retries must happen before a streamed token is emitted. Once a token has reached a client, retrying could duplicate output; the API therefore emits a structured server error for that case. Request cancellation propagates through the async stream and stops mock/backend work at its next yield point.

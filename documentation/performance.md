# Performance experiments

Start the service, run `make benchmark`, and save JSON output with `--json results.json`. Repeat with `MINI_MAX_BATCH_SIZE=1` and then with a larger value. Compare p50/p95 latency and requests per second at the same concurrency. The mock backend is useful for scheduler overhead and routing comparisons; use the optional Transformers backend on MPS/CUDA for model-runtime measurements.

Report TTFT, end-to-end latency, tokens per second, batch-size distribution, cache utilization, and worker health. Do not compare results across different prompt lengths or model/device settings without recording those variables.

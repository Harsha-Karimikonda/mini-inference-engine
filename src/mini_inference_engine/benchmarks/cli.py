import argparse
import asyncio
import json
import statistics
import time
import httpx


async def run(args):
    latencies = []
    semaphore = asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient(timeout=args.timeout) as client:
        async def one(index):
            async with semaphore:
                started = time.perf_counter()
                response = await client.post(args.endpoint.rstrip("/") + "/v1/completions", json={"model": "mock", "prompt": f"benchmark request {index}", "max_tokens": args.max_tokens, "stream": args.stream})
                response.raise_for_status()
                latencies.append(time.perf_counter() - started)
        await asyncio.gather(*(one(i) for i in range(args.requests)))
    result = {"requests": len(latencies), "concurrency": args.concurrency, "latency_p50_ms": statistics.median(latencies) * 1000, "latency_p95_ms": sorted(latencies)[max(0, int(len(latencies) * .95) - 1)] * 1000, "requests_per_second": len(latencies) / sum(latencies) * args.concurrency}
    print(json.dumps(result, indent=2))
    if args.json:
        with open(args.json, "w", encoding="utf8") as handle:
            json.dump(result, handle, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Benchmark Mini-Together")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--json")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""dfastllm Benchmark Suite

Run benchmarks against a dfastllm server to measure performance.

Usage:
    # Start server first:
    dfastllm --model gpt2 --port 8000

    # Run benchmark:
    python benchmarks/run_benchmark.py --url http://localhost:8000

    # With options:
    python benchmarks/run_benchmark.py \
        --url http://localhost:8000 \
        --requests 500 \
        --concurrency 20 \
        --max-tokens 128
"""

import asyncio
import time
import statistics
import argparse
import json
import sys
from dataclasses import dataclass, asdict
from typing import List, Optional
from datetime import datetime

try:
    import httpx
except ImportError:
    print("Error: httpx is required. Install with: pip install httpx")
    sys.exit(1)


@dataclass
class BenchmarkResult:
    """Benchmark result data."""
    timestamp: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_time_s: float
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    p50_latency_ms: float
    p90_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_rps: float
    tokens_per_second: float
    total_prompt_tokens: int
    total_completion_tokens: int
    
    def to_dict(self) -> dict:
        return asdict(self)


async def run_single_request(
    client: httpx.AsyncClient,
    url: str,
    prompt: str,
    max_tokens: int,
    semaphore: asyncio.Semaphore,
) -> tuple:
    """Run a single request and return (latency_ms, prompt_tokens, completion_tokens, success)."""
    async with semaphore:
        start = time.perf_counter()
        try:
            response = await client.post(
                f"{url}/v1/completions",
                json={
                    "model": "default",
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                },
                timeout=120.0,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            
            if response.status_code == 200:
                data = response.json()
                usage = data.get("usage", {})
                return (
                    elapsed_ms,
                    usage.get("prompt_tokens", 0),
                    usage.get("completion_tokens", 0),
                    True,
                )
            else:
                return (elapsed_ms, 0, 0, False)
                
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return (elapsed_ms, 0, 0, False)


async def run_benchmark(
    url: str,
    num_requests: int = 100,
    concurrency: int = 10,
    prompt: str = "Once upon a time in a land far away,",
    max_tokens: int = 64,
    warmup_requests: int = 5,
) -> BenchmarkResult:
    """Run benchmark against dfastllm server."""
    
    print(f"\n{'='*60}")
    print("dfastllm Benchmark")
    print(f"{'='*60}")
    print(f"URL:          {url}")
    print(f"Requests:     {num_requests}")
    print(f"Concurrency:  {concurrency}")
    print(f"Max Tokens:   {max_tokens}")
    print(f"Warmup:       {warmup_requests} requests")
    print(f"{'='*60}\n")
    
    # Check server health
    async with httpx.AsyncClient() as client:
        try:
            health = await client.get(f"{url}/health", timeout=10.0)
            if health.status_code != 200:
                print(f"Warning: Health check returned {health.status_code}")
        except Exception as e:
            print(f"Error: Cannot connect to server at {url}")
            print(f"       {e}")
            sys.exit(1)
    
    semaphore = asyncio.Semaphore(concurrency)
    
    # Warmup
    if warmup_requests > 0:
        print(f"Running {warmup_requests} warmup requests...")
        async with httpx.AsyncClient() as client:
            warmup_tasks = [
                run_single_request(client, url, prompt, max_tokens, semaphore)
                for _ in range(warmup_requests)
            ]
            await asyncio.gather(*warmup_tasks)
        print("Warmup complete.\n")
    
    # Run benchmark
    print(f"Running {num_requests} benchmark requests...")
    start_time = time.perf_counter()
    
    async with httpx.AsyncClient() as client:
        tasks = [
            run_single_request(client, url, prompt, max_tokens, semaphore)
            for _ in range(num_requests)
        ]
        results = await asyncio.gather(*tasks)
    
    total_time = time.perf_counter() - start_time
    
    # Process results
    latencies = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    failures = 0
    
    for latency_ms, prompt_tokens, completion_tokens, success in results:
        if success:
            latencies.append(latency_ms)
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens
        else:
            failures += 1
    
    if not latencies:
        print("Error: All requests failed!")
        sys.exit(1)
    
    latencies.sort()
    n = len(latencies)
    
    return BenchmarkResult(
        timestamp=datetime.now().isoformat(),
        total_requests=num_requests,
        successful_requests=n,
        failed_requests=failures,
        total_time_s=total_time,
        avg_latency_ms=statistics.mean(latencies),
        min_latency_ms=min(latencies),
        max_latency_ms=max(latencies),
        p50_latency_ms=latencies[n // 2],
        p90_latency_ms=latencies[int(n * 0.90)],
        p95_latency_ms=latencies[int(n * 0.95)],
        p99_latency_ms=latencies[int(n * 0.99)] if n >= 100 else latencies[-1],
        throughput_rps=n / total_time,
        tokens_per_second=total_completion_tokens / total_time,
        total_prompt_tokens=total_prompt_tokens,
        total_completion_tokens=total_completion_tokens,
    )


def print_results(result: BenchmarkResult) -> None:
    """Print benchmark results in a formatted table."""
    print(f"\n{'='*60}")
    print("BENCHMARK RESULTS")
    print(f"{'='*60}")
    print(f"Timestamp:            {result.timestamp}")
    print(f"\n{'─'*60}")
    print("Requests")
    print(f"{'─'*60}")
    print(f"  Total:              {result.total_requests}")
    print(f"  Successful:         {result.successful_requests}")
    print(f"  Failed:             {result.failed_requests}")
    print(f"  Success Rate:       {result.successful_requests/result.total_requests*100:.1f}%")
    print(f"\n{'─'*60}")
    print("Throughput")
    print(f"{'─'*60}")
    print(f"  Total Time:         {result.total_time_s:.2f}s")
    print(f"  Requests/sec:       {result.throughput_rps:.2f}")
    print(f"  Tokens/sec:         {result.tokens_per_second:.2f}")
    print(f"\n{'─'*60}")
    print("Latency (ms)")
    print(f"{'─'*60}")
    print(f"  Min:                {result.min_latency_ms:.2f}")
    print(f"  Average:            {result.avg_latency_ms:.2f}")
    print(f"  Max:                {result.max_latency_ms:.2f}")
    print(f"  P50:                {result.p50_latency_ms:.2f}")
    print(f"  P90:                {result.p90_latency_ms:.2f}")
    print(f"  P95:                {result.p95_latency_ms:.2f}")
    print(f"  P99:                {result.p99_latency_ms:.2f}")
    print(f"\n{'─'*60}")
    print("Tokens")
    print(f"{'─'*60}")
    print(f"  Total Prompt:       {result.total_prompt_tokens}")
    print(f"  Total Completion:   {result.total_completion_tokens}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="dfastllm Benchmark Suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000",
        help="dfastllm server URL",
    )
    parser.add_argument(
        "--requests", "-n",
        type=int,
        default=100,
        help="Number of requests to run",
    )
    parser.add_argument(
        "--concurrency", "-c",
        type=int,
        default=10,
        help="Number of concurrent requests",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64,
        help="Maximum tokens to generate per request",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Once upon a time in a land far away,",
        help="Prompt to use for generation",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of warmup requests",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file for results (JSON)",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Use chat completions endpoint instead",
    )
    
    args = parser.parse_args()
    
    result = asyncio.run(run_benchmark(
        url=args.url,
        num_requests=args.requests,
        concurrency=args.concurrency,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        warmup_requests=args.warmup,
    ))
    
    print_results(result)
    
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()


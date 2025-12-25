#!/usr/bin/env python3
"""Benchmark Time-To-First-Token (TTFT) for dfastllm.

Measures first token latency and compares streaming vs non-streaming performance.
"""

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass, asdict
from typing import List, Optional
import httpx


@dataclass
class TTFTResult:
    """Result from a single TTFT measurement."""
    ttft_ms: float  # Time to first token in milliseconds
    total_time_ms: float  # Total generation time
    first_chunk_tokens: int  # Tokens in first chunk
    total_tokens: int  # Total tokens generated
    streaming: bool  # Whether streaming was used


@dataclass
class TTFTBenchmarkSummary:
    """Summary of TTFT benchmark results."""
    model: str
    prompt_length: int
    output_tokens: int
    num_requests: int
    
    # TTFT metrics
    ttft_mean_ms: float
    ttft_p50_ms: float
    ttft_p90_ms: float
    ttft_p99_ms: float
    ttft_min_ms: float
    ttft_max_ms: float
    ttft_std_ms: float
    
    # Total time metrics
    total_time_mean_ms: float
    total_time_p50_ms: float
    total_time_p90_ms: float
    
    # Throughput
    tokens_per_second: float


def percentile(data: List[float], p: float) -> float:
    """Calculate the p-th percentile of data."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100
    f = int(k)
    c = f + 1 if f + 1 < len(sorted_data) else f
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f]) if f != c else sorted_data[f]


async def measure_ttft_streaming(
    client: httpx.AsyncClient,
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
) -> TTFTResult:
    """Measure TTFT using streaming endpoint."""
    start_time = time.perf_counter()
    first_chunk_time = None
    first_chunk_tokens = 0
    total_tokens = 0
    
    async with client.stream(
        "POST",
        f"{url}/v1/completions",
        json={
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "stream": True,
        },
        timeout=120.0,
    ) as response:
        async for line in response.aiter_lines():
            if line.startswith("data: "):
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    if chunk.get("choices"):
                        text = chunk["choices"][0].get("text", "")
                        # Rough token estimate (word-based)
                        chunk_tokens = max(1, len(text.split()))
                        total_tokens += chunk_tokens
                        
                        if first_chunk_time is None:
                            first_chunk_time = time.perf_counter()
                            first_chunk_tokens = chunk_tokens
                except json.JSONDecodeError:
                    continue
    
    end_time = time.perf_counter()
    
    ttft = (first_chunk_time - start_time) * 1000 if first_chunk_time else (end_time - start_time) * 1000
    total_time = (end_time - start_time) * 1000
    
    return TTFTResult(
        ttft_ms=ttft,
        total_time_ms=total_time,
        first_chunk_tokens=first_chunk_tokens,
        total_tokens=total_tokens,
        streaming=True,
    )


async def measure_ttft_non_streaming(
    client: httpx.AsyncClient,
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
) -> TTFTResult:
    """Measure latency for non-streaming endpoint (TTFT = total time)."""
    start_time = time.perf_counter()
    
    response = await client.post(
        f"{url}/v1/completions",
        json={
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "stream": False,
        },
        timeout=120.0,
    )
    
    end_time = time.perf_counter()
    
    data = response.json()
    text = data.get("choices", [{}])[0].get("text", "")
    # Rough token estimate
    total_tokens = max(1, len(text.split()))
    
    total_time = (end_time - start_time) * 1000
    
    return TTFTResult(
        ttft_ms=total_time,  # For non-streaming, TTFT = total time
        total_time_ms=total_time,
        first_chunk_tokens=total_tokens,
        total_tokens=total_tokens,
        streaming=False,
    )


async def run_benchmark(
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    num_requests: int,
    use_streaming: bool,
    warmup_requests: int = 3,
) -> TTFTBenchmarkSummary:
    """Run TTFT benchmark."""
    results: List[TTFTResult] = []
    
    async with httpx.AsyncClient() as client:
        # Warmup
        print(f"  Running {warmup_requests} warmup requests...")
        for _ in range(warmup_requests):
            if use_streaming:
                await measure_ttft_streaming(client, url, model, prompt, max_tokens)
            else:
                await measure_ttft_non_streaming(client, url, model, prompt, max_tokens)
        
        # Actual measurements
        print(f"  Running {num_requests} benchmark requests...")
        for i in range(num_requests):
            if use_streaming:
                result = await measure_ttft_streaming(client, url, model, prompt, max_tokens)
            else:
                result = await measure_ttft_non_streaming(client, url, model, prompt, max_tokens)
            results.append(result)
            
            if (i + 1) % 10 == 0:
                print(f"    Completed {i + 1}/{num_requests} requests")
    
    # Calculate statistics
    ttfts = [r.ttft_ms for r in results]
    total_times = [r.total_time_ms for r in results]
    total_tokens = sum(r.total_tokens for r in results)
    total_time_s = sum(r.total_time_ms for r in results) / 1000
    
    return TTFTBenchmarkSummary(
        model=model,
        prompt_length=len(prompt.split()),
        output_tokens=max_tokens,
        num_requests=num_requests,
        ttft_mean_ms=statistics.mean(ttfts),
        ttft_p50_ms=percentile(ttfts, 50),
        ttft_p90_ms=percentile(ttfts, 90),
        ttft_p99_ms=percentile(ttfts, 99),
        ttft_min_ms=min(ttfts),
        ttft_max_ms=max(ttfts),
        ttft_std_ms=statistics.stdev(ttfts) if len(ttfts) > 1 else 0,
        total_time_mean_ms=statistics.mean(total_times),
        total_time_p50_ms=percentile(total_times, 50),
        total_time_p90_ms=percentile(total_times, 90),
        tokens_per_second=total_tokens / total_time_s if total_time_s > 0 else 0,
    )


def print_summary(summary: TTFTBenchmarkSummary, streaming: bool):
    """Print benchmark summary in a nice format."""
    mode = "Streaming" if streaming else "Non-Streaming"
    
    print(f"\n{'='*60}")
    print(f"TTFT Benchmark Results ({mode})")
    print(f"{'='*60}")
    print(f"Model: {summary.model}")
    print(f"Prompt Length: ~{summary.prompt_length} words")
    print(f"Output Tokens: {summary.output_tokens}")
    print(f"Requests: {summary.num_requests}")
    print(f"{'-'*60}")
    print("Time to First Token (TTFT):")
    print(f"  Mean:   {summary.ttft_mean_ms:>10.1f} ms")
    print(f"  P50:    {summary.ttft_p50_ms:>10.1f} ms")
    print(f"  P90:    {summary.ttft_p90_ms:>10.1f} ms")
    print(f"  P99:    {summary.ttft_p99_ms:>10.1f} ms")
    print(f"  Min:    {summary.ttft_min_ms:>10.1f} ms")
    print(f"  Max:    {summary.ttft_max_ms:>10.1f} ms")
    print(f"  Std:    {summary.ttft_std_ms:>10.1f} ms")
    print(f"{'-'*60}")
    print("Total Generation Time:")
    print(f"  Mean:   {summary.total_time_mean_ms:>10.1f} ms")
    print(f"  P50:    {summary.total_time_p50_ms:>10.1f} ms")
    print(f"  P90:    {summary.total_time_p90_ms:>10.1f} ms")
    print(f"{'-'*60}")
    print(f"Throughput: {summary.tokens_per_second:.1f} tokens/sec")
    print(f"{'='*60}")


async def main():
    parser = argparse.ArgumentParser(description="Benchmark TTFT for dfastllm")
    parser.add_argument("--url", default="http://localhost:8000", help="API URL")
    parser.add_argument("--model", default="phi-2", help="Model name")
    parser.add_argument("--prompt", default="Write a detailed explanation of how artificial intelligence works.",
                        help="Prompt to use")
    parser.add_argument("--max-tokens", type=int, default=64, help="Max tokens to generate")
    parser.add_argument("--requests", type=int, default=20, help="Number of requests")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup requests")
    parser.add_argument("--compare", action="store_true", help="Compare streaming vs non-streaming")
    parser.add_argument("--output", help="Output JSON file")
    
    args = parser.parse_args()
    
    print(f"\n🚀 TTFT Benchmark for dfastllm")
    print(f"   URL: {args.url}")
    print(f"   Model: {args.model}")
    print(f"   Max Tokens: {args.max_tokens}")
    print(f"   Requests: {args.requests}")
    
    results = {}
    
    # Streaming benchmark
    print(f"\n📊 Running streaming benchmark...")
    streaming_summary = await run_benchmark(
        url=args.url,
        model=args.model,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        num_requests=args.requests,
        use_streaming=True,
        warmup_requests=args.warmup,
    )
    print_summary(streaming_summary, streaming=True)
    results["streaming"] = asdict(streaming_summary)
    
    if args.compare:
        # Non-streaming benchmark
        print(f"\n📊 Running non-streaming benchmark...")
        non_streaming_summary = await run_benchmark(
            url=args.url,
            model=args.model,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            num_requests=args.requests,
            use_streaming=False,
            warmup_requests=args.warmup,
        )
        print_summary(non_streaming_summary, streaming=False)
        results["non_streaming"] = asdict(non_streaming_summary)
        
        # Comparison
        print(f"\n{'='*60}")
        print("📈 COMPARISON")
        print(f"{'='*60}")
        improvement = ((non_streaming_summary.ttft_mean_ms - streaming_summary.ttft_mean_ms) 
                      / non_streaming_summary.ttft_mean_ms * 100)
        print(f"Streaming TTFT:     {streaming_summary.ttft_mean_ms:.1f} ms")
        print(f"Non-Streaming TTFT: {non_streaming_summary.ttft_mean_ms:.1f} ms")
        print(f"TTFT Improvement:   {improvement:+.1f}%")
        print(f"{'='*60}")
        
        results["comparison"] = {
            "ttft_improvement_pct": improvement,
            "streaming_ttft_ms": streaming_summary.ttft_mean_ms,
            "non_streaming_ttft_ms": non_streaming_summary.ttft_mean_ms,
        }
    
    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())


#!/usr/bin/env python
"""Simple benchmarking script for dfastllm Serving.

This script measures throughput and latency of the dfastllm server.

Requirements:
    pip install requests numpy

Usage:
    # Start the dfastllm server first:
    # python -m dfastllm.entrypoints.openai.api_server --model GSAI-ML/LLaDA-8B-Instruct

    # Basic benchmark:
    python examples/benchmark.py

    # With options:
    python examples/benchmark.py --num-requests 100 --concurrent 10
"""

import argparse
import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

import requests


def create_request(
    base_url: str,
    prompt: str,
    max_tokens: int,
    model: str,
) -> Dict[str, Any]:
    """Send a single completion request and measure latency."""
    start_time = time.time()
    
    try:
        response = requests.post(
            f"{base_url}/v1/completions",
            json={
                "model": model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.7,
            },
            headers={"Content-Type": "application/json"},
            timeout=120,
        )
        
        end_time = time.time()
        latency = end_time - start_time
        
        if response.status_code == 200:
            data = response.json()
            usage = data.get("usage", {})
            return {
                "success": True,
                "latency": latency,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "parallel_tokens": data.get("parallel_tokens_decoded", 0),
            }
        else:
            return {
                "success": False,
                "latency": latency,
                "error": response.text,
            }
    except Exception as e:
        return {
            "success": False,
            "latency": time.time() - start_time,
            "error": str(e),
        }


def run_benchmark(
    base_url: str,
    model: str,
    num_requests: int,
    concurrent: int,
    prompt: str,
    max_tokens: int,
) -> Dict[str, Any]:
    """Run the benchmark."""
    print(f"Running benchmark with {num_requests} requests ({concurrent} concurrent)...")
    
    results: List[Dict[str, Any]] = []
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=concurrent) as executor:
        futures = [
            executor.submit(create_request, base_url, prompt, max_tokens, model)
            for _ in range(num_requests)
        ]
        
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            results.append(result)
            if (i + 1) % 10 == 0:
                print(f"  Completed {i + 1}/{num_requests} requests")
    
    total_time = time.time() - start_time
    
    # Calculate statistics
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    
    latencies = [r["latency"] for r in successful]
    total_prompt_tokens = sum(r["prompt_tokens"] for r in successful)
    total_completion_tokens = sum(r["completion_tokens"] for r in successful)
    total_parallel_tokens = sum(r.get("parallel_tokens", 0) for r in successful)
    
    stats = {
        "total_requests": num_requests,
        "successful_requests": len(successful),
        "failed_requests": len(failed),
        "total_time_seconds": total_time,
        "requests_per_second": num_requests / total_time,
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "parallel_tokens_decoded": total_parallel_tokens,
        "tokens_per_second": total_completion_tokens / total_time if total_time > 0 else 0,
    }
    
    if latencies:
        stats["latency_mean"] = statistics.mean(latencies)
        stats["latency_median"] = statistics.median(latencies)
        stats["latency_p90"] = sorted(latencies)[int(len(latencies) * 0.9)]
        stats["latency_p99"] = sorted(latencies)[int(len(latencies) * 0.99)]
        stats["latency_min"] = min(latencies)
        stats["latency_max"] = max(latencies)
    
    return stats


def print_results(stats: Dict[str, Any]):
    """Print benchmark results."""
    print("\n" + "=" * 60)
    print("Benchmark Results")
    print("=" * 60)
    
    print(f"\nRequests:")
    print(f"  Total:      {stats['total_requests']}")
    print(f"  Successful: {stats['successful_requests']}")
    print(f"  Failed:     {stats['failed_requests']}")
    
    print(f"\nThroughput:")
    print(f"  Total time:           {stats['total_time_seconds']:.2f} seconds")
    print(f"  Requests/second:      {stats['requests_per_second']:.2f}")
    print(f"  Tokens/second:        {stats['tokens_per_second']:.2f}")
    
    print(f"\nTokens:")
    print(f"  Prompt tokens:        {stats['prompt_tokens']}")
    print(f"  Completion tokens:    {stats['completion_tokens']}")
    print(f"  Parallel decoded:     {stats['parallel_tokens_decoded']}")
    
    if "latency_mean" in stats:
        print(f"\nLatency (seconds):")
        print(f"  Mean:    {stats['latency_mean']:.3f}")
        print(f"  Median:  {stats['latency_median']:.3f}")
        print(f"  P90:     {stats['latency_p90']:.3f}")
        print(f"  P99:     {stats['latency_p99']:.3f}")
        print(f"  Min:     {stats['latency_min']:.3f}")
        print(f"  Max:     {stats['latency_max']:.3f}")
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="dfastllm Benchmark")
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8000",
        help="Base URL of the dfastllm server",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="GSAI-ML/LLaDA-8B-Instruct",
        help="Model to use",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=50,
        help="Number of requests to send",
    )
    parser.add_argument(
        "--concurrent",
        type=int,
        default=5,
        help="Number of concurrent requests",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Maximum tokens per request",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Explain the concept of machine learning in simple terms:",
        help="Prompt to use",
    )
    
    args = parser.parse_args()
    
    # Check server health
    print(f"Checking server health at {args.base_url}...")
    try:
        response = requests.get(f"{args.base_url}/health", timeout=5)
        if response.status_code != 200:
            print(f"Server not healthy: {response.status_code}")
            return
    except requests.ConnectionError:
        print("Cannot connect to server. Is dfastllm running?")
        return
    
    print("Server is healthy!")
    
    # Run benchmark
    stats = run_benchmark(
        base_url=args.base_url,
        model=args.model,
        num_requests=args.num_requests,
        concurrent=args.concurrent,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
    )
    
    # Print results
    print_results(stats)


if __name__ == "__main__":
    main()

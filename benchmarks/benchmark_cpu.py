#!/usr/bin/env python3
"""CPU Performance Benchmark for dfastllm.

Tests dfastllm performance on CPU with various configurations.
Measures TTFT, throughput, and memory usage.
"""

import argparse
import gc
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Optional

# Force CPU mode
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch


@dataclass
class BenchmarkResult:
    """Benchmark result."""
    prompt: str
    tokens_generated: int
    time_ms: float
    ttft_ms: float
    tokens_per_second: float
    memory_mb: float


def get_memory_usage() -> float:
    """Get current memory usage in MB."""
    import psutil
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def benchmark_generation(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 32,
    temperature: float = 0.0,
) -> BenchmarkResult:
    """Run a single generation benchmark."""
    from dfastllm.engine.diffusion_generator import (
        DiffusionGenerator,
        DiffusionConfig,
        GenerationMode,
    )
    
    # Create generator
    config = DiffusionConfig(
        mode=GenerationMode.FAST,
        min_steps=4,
        max_steps=16,
    )
    generator = DiffusionGenerator(model, tokenizer, config=config)
    
    # Encode prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    prompt_ids = inputs["input_ids"]
    
    # Warmup
    gc.collect()
    start_mem = get_memory_usage()
    
    # Generate with streaming to measure TTFT
    start_time = time.perf_counter()
    ttft = None
    
    for chunk in generator.generate_stream(prompt_ids, max_tokens=max_tokens, temperature=temperature):
        if ttft is None and chunk.tokens_revealed > 0:
            ttft = (time.perf_counter() - start_time) * 1000
        if chunk.is_final:
            output_text = chunk.text
            break
    
    end_time = time.perf_counter()
    end_mem = get_memory_usage()
    
    elapsed_ms = (end_time - start_time) * 1000
    
    return BenchmarkResult(
        prompt=prompt,
        tokens_generated=max_tokens,
        time_ms=elapsed_ms,
        ttft_ms=ttft or elapsed_ms,
        tokens_per_second=(max_tokens / elapsed_ms * 1000) if elapsed_ms > 0 else 0,
        memory_mb=end_mem - start_mem,
    )


def run_cpu_benchmark(
    model_name: str = "microsoft/phi-2",
    num_runs: int = 3,
    max_tokens: int = 32,
):
    """Run full CPU benchmark suite."""
    print("=" * 60)
    print("🖥️  dfastllm CPU Benchmark")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Device: CPU (threads: {torch.get_num_threads()})")
    print(f"Max Tokens: {max_tokens}")
    print(f"Runs: {num_runs}")
    print("=" * 60)
    print()
    
    # Load model
    print("📦 Loading model...")
    start_load = time.time()
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # CPU uses float32
        device_map="cpu",
        trust_remote_code=True,
    )
    model.eval()
    
    load_time = time.time() - start_load
    print(f"✅ Model loaded in {load_time:.1f}s")
    print(f"   Memory: {get_memory_usage():.0f} MB")
    print()
    
    # Test prompts
    prompts = [
        "Hello, how are you?",
        "The quick brown fox",
        "Machine learning is",
        "Python programming involves",
    ]
    
    results: List[BenchmarkResult] = []
    
    print("🔄 Running benchmarks...")
    print()
    
    for i, prompt in enumerate(prompts):
        print(f"  Prompt {i+1}/{len(prompts)}: '{prompt[:30]}...'")
        
        run_results = []
        for run in range(num_runs):
            result = benchmark_generation(model, tokenizer, prompt, max_tokens)
            run_results.append(result)
            print(f"    Run {run+1}: {result.tokens_per_second:.1f} tok/s, TTFT: {result.ttft_ms:.0f}ms")
        
        # Average the runs
        avg_result = BenchmarkResult(
            prompt=prompt,
            tokens_generated=max_tokens,
            time_ms=sum(r.time_ms for r in run_results) / len(run_results),
            ttft_ms=sum(r.ttft_ms for r in run_results) / len(run_results),
            tokens_per_second=sum(r.tokens_per_second for r in run_results) / len(run_results),
            memory_mb=sum(r.memory_mb for r in run_results) / len(run_results),
        )
        results.append(avg_result)
        print()
    
    # Summary
    print("=" * 60)
    print("📊 BENCHMARK RESULTS")
    print("=" * 60)
    print()
    
    avg_ttft = sum(r.ttft_ms for r in results) / len(results)
    avg_throughput = sum(r.tokens_per_second for r in results) / len(results)
    avg_time = sum(r.time_ms for r in results) / len(results)
    
    print(f"{'Metric':<25} {'Value':>15}")
    print("-" * 40)
    print(f"{'Avg TTFT (ms)':<25} {avg_ttft:>15.1f}")
    print(f"{'Avg Throughput (tok/s)':<25} {avg_throughput:>15.1f}")
    print(f"{'Avg Time (ms)':<25} {avg_time:>15.1f}")
    print(f"{'Memory Usage (MB)':<25} {get_memory_usage():>15.0f}")
    print()
    
    # Per-prompt results
    print("Per-Prompt Results:")
    print("-" * 60)
    for r in results:
        print(f"  '{r.prompt[:25]:25s}': {r.tokens_per_second:>6.1f} tok/s, TTFT: {r.ttft_ms:>6.0f}ms")
    
    print()
    print("=" * 60)
    print("✅ Benchmark Complete")
    print("=" * 60)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="dfastllm CPU Benchmark")
    parser.add_argument("--model", default="microsoft/phi-2", help="Model to benchmark")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per prompt")
    parser.add_argument("--max-tokens", type=int, default=32, help="Max tokens to generate")
    parser.add_argument("--threads", type=int, default=None, help="CPU threads to use")
    
    args = parser.parse_args()
    
    # Set CPU threads
    if args.threads:
        torch.set_num_threads(args.threads)
        print(f"Using {args.threads} CPU threads")
    
    run_cpu_benchmark(
        model_name=args.model,
        num_runs=args.runs,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()


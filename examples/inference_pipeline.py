#!/usr/bin/env python3
"""End-to-End Inference Pipeline Example for vdiff.

This script demonstrates the complete vdiff inference pipeline with:
1. Standard diffusion generation
2. APD (Adaptive Parallel Decoding) generation
3. Comparison of both methods

Usage:
    # With a real diffusion model (LLaDA):
    python examples/inference_pipeline.py --model GSAI-ML/LLaDA-8B-Instruct
    
    # With a regular model (for testing):
    python examples/inference_pipeline.py --model gpt2
    
    # Enable APD:
    python examples/inference_pipeline.py --model gpt2 --enable-apd
"""

import argparse
import time
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vdiff.config import VDiffConfig
from vdiff.engine import VDiffEngine, SamplingParams


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def print_stats(stats: dict, prefix: str = ""):
    """Print statistics in a formatted way."""
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"{prefix}{key}:")
            print_stats(value, prefix + "  ")
        elif isinstance(value, float):
            print(f"{prefix}{key}: {value:.4f}")
        else:
            print(f"{prefix}{key}: {value}")


def run_inference(
    engine: VDiffEngine,
    prompt: str,
    max_tokens: int = 32,
    temperature: float = 0.0,
) -> dict:
    """Run inference and return results with timing."""
    
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.95,
    )
    
    start_time = time.time()
    output = engine.generate(prompt, sampling_params)
    elapsed_time = time.time() - start_time
    
    return {
        "prompt": prompt,
        "generated_text": output.outputs[0].text,
        "prompt_tokens": output.metrics.prompt_tokens if output.metrics else 0,
        "generated_tokens": output.metrics.generated_tokens if output.metrics else 0,
        "elapsed_time_ms": elapsed_time * 1000,
        "tokens_per_second": (output.metrics.generated_tokens / elapsed_time) if output.metrics and elapsed_time > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="vdiff Inference Pipeline Example")
    parser.add_argument("--model", type=str, default="gpt2", help="Model name or path")
    parser.add_argument("--max-tokens", type=int, default=32, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--enable-apd", action="store_true", help="Enable APD decoding")
    parser.add_argument("--apd-max-parallel", type=int, default=8, help="APD max parallel tokens")
    parser.add_argument("--diffusion-steps", type=int, default=32, help="Diffusion steps")
    parser.add_argument("--dtype", type=str, default="auto", help="Model dtype")
    parser.add_argument("--trust-remote-code", action="store_true", help="Trust remote code")
    args = parser.parse_args()
    
    print_header("vdiff Inference Pipeline")
    print(f"Model: {args.model}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Temperature: {args.temperature}")
    print(f"APD enabled: {args.enable_apd}")
    print(f"Diffusion steps: {args.diffusion_steps}")
    
    # Create configuration
    config = VDiffConfig(
        model=args.model,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        diffusion_steps=args.diffusion_steps,
        enable_apd=args.enable_apd,
        apd_max_parallel=args.apd_max_parallel,
    )
    
    # Initialize engine
    print_header("Initializing Engine")
    start_time = time.time()
    engine = VDiffEngine(config)
    init_time = time.time() - start_time
    print(f"Engine initialized in {init_time:.2f}s")
    print(f"Device: {engine._device}")
    print(f"Is diffusion model: {engine._is_diffusion_model}")
    
    # Test prompts
    prompts = [
        "Hello, how are you?",
        "What is the capital of France?",
        "Explain quantum computing in simple terms:",
    ]
    
    # Run inference
    print_header("Running Inference")
    results = []
    
    for i, prompt in enumerate(prompts):
        print(f"\n[{i+1}/{len(prompts)}] Prompt: {prompt[:50]}...")
        
        result = run_inference(
            engine=engine,
            prompt=prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        results.append(result)
        
        print(f"  Generated: {result['generated_text'][:100]}...")
        print(f"  Tokens: {result['generated_tokens']}")
        print(f"  Time: {result['elapsed_time_ms']:.1f}ms")
        print(f"  Speed: {result['tokens_per_second']:.1f} tok/s")
    
    # Print summary
    print_header("Summary")
    
    total_tokens = sum(r["generated_tokens"] for r in results)
    total_time = sum(r["elapsed_time_ms"] for r in results)
    avg_speed = total_tokens / (total_time / 1000) if total_time > 0 else 0
    
    print(f"Total requests: {len(results)}")
    print(f"Total tokens generated: {total_tokens}")
    print(f"Total time: {total_time:.1f}ms")
    print(f"Average speed: {avg_speed:.1f} tokens/second")
    
    # Print engine stats
    print_header("Engine Statistics")
    print_stats(engine.get_stats())
    
    # Shutdown
    print_header("Shutting Down")
    engine.shutdown()
    print("Done!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Hybrid Diffusion-AR Generation Demo.

This example demonstrates the hybrid generation approach combining:
- Diffusion LLM for fast parallel token drafting
- Autoregressive LLM for quality verification

Based on research papers:
- DEER: Draft with Diffusion, Verify with AR (https://czc726.github.io/DEER/)
- DiffuSpec: Speculative Decoding with Diffusion (arxiv:2510.02358)
- SpecDiff: Speculative Diffusion Decoding (NAACL 2025)

Benefits:
- 2-7x speedup over pure AR generation
- Maintains AR-level output quality
- No model retraining required

Usage:
    # Basic hybrid mode
    python hybrid_demo.py --model GSAI-ML/LLaDA-8B-Instruct
    
    # With custom AR verifier
    python hybrid_demo.py \
        --model GSAI-ML/LLaDA-8B-Instruct \
        --ar-model TinyLlama/TinyLlama-1.1B-Chat-v1.0

    # Compare modes
    python hybrid_demo.py --compare
"""

import argparse
import time
import sys
from typing import Optional

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Install with: pip install torch")
    sys.exit(1)


def load_models(
    diffusion_model_name: str,
    ar_model_name: Optional[str] = None,
    device: str = "auto",
):
    """Load diffusion and AR models."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"Loading diffusion model: {diffusion_model_name}")
    
    # Determine device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        diffusion_model_name,
        trust_remote_code=True,
    )
    
    # Load diffusion model
    diffusion_model = AutoModelForCausalLM.from_pretrained(
        diffusion_model_name,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    
    if device != "cuda":
        diffusion_model = diffusion_model.to(device)
    
    diffusion_model.eval()
    print(f"  Loaded on {device}")
    
    # Load AR verifier if specified
    ar_model = None
    if ar_model_name:
        print(f"Loading AR verifier: {ar_model_name}")
        ar_model = AutoModelForCausalLM.from_pretrained(
            ar_model_name,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
        )
        if device != "cuda":
            ar_model = ar_model.to(device)
        ar_model.eval()
        print(f"  Loaded on {device}")
    
    return diffusion_model, ar_model, tokenizer


def demo_hybrid_engine():
    """Demonstrate hybrid engine usage."""
    from dfastllm.engine import (
        HybridEngine,
        HybridConfig,
        HybridMode,
        create_hybrid_engine,
    )
    
    print("\n" + "=" * 60)
    print("Hybrid Diffusion-AR Engine Demo")
    print("=" * 60)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Hybrid generation demo")
    parser.add_argument(
        "--model",
        default="GSAI-ML/LLaDA-8B-Instruct",
        help="Diffusion model name/path",
    )
    parser.add_argument(
        "--ar-model",
        default=None,
        help="AR verifier model name/path (optional)",
    )
    parser.add_argument(
        "--mode",
        choices=["deer", "spec_diff", "semi_ar"],
        default="deer",
        help="Hybrid mode",
    )
    parser.add_argument(
        "--prompt",
        default="Explain the difference between autoregressive and diffusion language models in 3 sentences.",
        help="Input prompt",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--draft-size",
        type=int,
        default=8,
        help="Tokens per draft",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare hybrid vs pure diffusion",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device (auto, cuda, cpu)",
    )
    
    args = parser.parse_args()
    
    # Load models
    diffusion_model, ar_model, tokenizer = load_models(
        args.model,
        args.ar_model,
        args.device,
    )
    
    # Get mask token ID
    mask_id = getattr(tokenizer, 'mask_token_id', 126336)
    
    # Create hybrid config
    config = HybridConfig(
        enabled=True,
        mode=HybridMode(args.mode),
        draft_block_size=args.draft_size,
        acceptance_threshold=0.3,
        adaptive_draft_length=True,
        log_stats=True,
    )
    
    # Create hybrid engine
    print(f"\nCreating hybrid engine (mode={args.mode})...")
    engine = create_hybrid_engine(
        diffusion_model=diffusion_model,
        ar_model=ar_model,
        tokenizer=tokenizer,
        config=config,
        mask_id=mask_id,
    )
    
    # Tokenize prompt
    print(f"\nPrompt: {args.prompt}")
    inputs = tokenizer(args.prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(diffusion_model.device)
    
    # Generate with hybrid engine
    print(f"\nGenerating {args.max_tokens} tokens with hybrid engine...")
    start_time = time.perf_counter()
    
    with torch.no_grad():
        output_ids = engine.generate(
            input_ids=input_ids,
            max_new_tokens=args.max_tokens,
        )
    
    hybrid_time = time.perf_counter() - start_time
    
    # Decode output
    output_text = tokenizer.decode(
        output_ids[0][input_ids.shape[1]:],
        skip_special_tokens=True,
    )
    
    print(f"\n{'=' * 40}")
    print(f"Generated text:")
    print(f"{'=' * 40}")
    print(output_text)
    print(f"\n{'=' * 40}")
    print(f"Generation time: {hybrid_time:.2f}s")
    print(f"Tokens/second: {args.max_tokens / hybrid_time:.1f}")
    
    # Print hybrid stats
    stats = engine.get_stats()
    print(f"\nHybrid Stats:")
    print(f"  - Drafts: {stats.get('total_drafts', 0)}")
    print(f"  - Acceptance rate: {stats.get('draft_acceptance_rate', 0):.1%}")
    print(f"  - Avg accepted length: {stats.get('avg_accepted_length', 0):.1f}")
    print(f"  - Speedup ratio: {stats.get('speedup_ratio', 1):.2f}x")
    
    # Compare with pure diffusion if requested
    if args.compare:
        print(f"\n{'=' * 60}")
        print("Comparison: Pure Diffusion vs Hybrid")
        print("=" * 60)
        
        from dfastllm.engine import diffusion_generate
        
        # Pure diffusion
        print("\nGenerating with pure diffusion...")
        start_time = time.perf_counter()
        
        with torch.no_grad():
            pure_output = diffusion_generate(
                model=diffusion_model,
                prompt=input_ids,
                steps=64,
                gen_length=args.max_tokens,
                block_length=min(32, args.max_tokens),
                mask_id=mask_id,
            )
        
        pure_time = time.perf_counter() - start_time
        
        pure_text = tokenizer.decode(
            pure_output[0][input_ids.shape[1]:],
            skip_special_tokens=True,
        )
        
        print(f"\n{'=' * 40}")
        print(f"Comparison Results:")
        print(f"{'=' * 40}")
        print(f"Hybrid time: {hybrid_time:.2f}s ({args.max_tokens / hybrid_time:.1f} tok/s)")
        print(f"Pure diffusion time: {pure_time:.2f}s ({args.max_tokens / pure_time:.1f} tok/s)")
        print(f"Speedup: {pure_time / hybrid_time:.2f}x")
        
        print(f"\nHybrid output: {output_text[:100]}...")
        print(f"Pure output: {pure_text[:100]}...")


def demo_standalone_hybrid():
    """Demonstrate standalone hybrid_generate function."""
    from dfastllm.engine import hybrid_generate
    
    print("\n" + "=" * 60)
    print("Standalone hybrid_generate Function Demo")
    print("=" * 60)
    
    # This is a simpler interface for quick usage
    print("""
Example usage:

    from dfastllm.engine import hybrid_generate
    
    output = hybrid_generate(
        diffusion_model=my_diffusion_model,
        prompt=input_ids,
        ar_model=my_ar_model,  # optional
        max_new_tokens=64,
        mode="deer",
        draft_size=8,
    )
    """)


def demo_full_engine():
    """Demonstrate full DFastLLMEngine with hybrid mode."""
    print("\n" + "=" * 60)
    print("Full DFastLLMEngine with Hybrid Mode")
    print("=" * 60)
    
    print("""
Example usage:

    from dfastllm.config import DFastLLMConfig
    from dfastllm.engine import DFastLLMEngine, SamplingParams
    
    # Configure with hybrid mode enabled
    config = DFastLLMConfig(
        model="GSAI-ML/LLaDA-8B-Instruct",
        
        # Enable hybrid mode
        enable_hybrid=True,
        hybrid_mode="deer",  # or "spec_diff", "semi_ar"
        
        # Optional: Add AR verifier for quality
        ar_verifier_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        
        # Hybrid settings
        hybrid_draft_size=8,
        hybrid_acceptance_threshold=0.3,
        hybrid_adaptive_draft=True,
    )
    
    # Create engine
    engine = DFastLLMEngine(config)
    
    # Generate
    params = SamplingParams(max_tokens=64)
    output = engine.generate("Hello, ", params)
    print(output.outputs[0].text)
    
    # Check hybrid stats
    stats = engine.get_stats()
    print(f"Hybrid speedup: {stats['hybrid']['speedup_ratio']}x")
    """)


def main():
    """Main entry point."""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--help-modes":
        print("""
Hybrid Modes:
=============

1. DEER (deer) - Draft with Diffusion, Verify with AR
   - Best for: General text generation
   - How it works: Diffusion generates k tokens in parallel,
     AR model verifies sequence coherence, accept longest prefix
   - Speedup: 2-7x
   - Reference: https://czc726.github.io/DEER/

2. SpecDiff (spec_diff) - Speculative Diffusion Decoding
   - Best for: Maximum parallelism
   - How it works: Multi-step diffusion drafting with
     causal-consistency path search for verification
   - Speedup: Up to 7.2x (claimed)
   - Reference: NAACL 2025

3. Semi-AR (semi_ar) - Semi-Autoregressive
   - Best for: Balance of speed and quality
   - How it works: Generate blocks via diffusion,
     refine low-confidence tokens with AR
   - Speedup: 2-3x
   - Reference: SSD-LM paper (ACL 2023)

Environment Variables:
======================
VDIFF_HYBRID_ENABLED=true       Enable hybrid mode
VDIFF_HYBRID_MODE=deer          Hybrid mode
VDIFF_AR_VERIFIER_MODEL=...     AR verifier model path
VDIFF_HYBRID_DRAFT_SIZE=8       Tokens per draft
VDIFF_HYBRID_THRESHOLD=0.3      Acceptance threshold
        """)
        return
    
    try:
        demo_hybrid_engine()
    except Exception as e:
        print(f"\nError: {e}")
        print("\nShowing usage examples instead:")
        demo_standalone_hybrid()
        demo_full_engine()


if __name__ == "__main__":
    main()

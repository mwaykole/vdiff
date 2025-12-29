#!/usr/bin/env python3
"""Local test script for dfastllm diffusion sampler.

Tests the optimized diffusion generation code with a small model (GPT-2).
Works on CPU without GPU.

Usage:
    python scripts/test_local.py
    python scripts/test_local.py --steps 8 --gen-length 16
    python scripts/test_local.py --model gpt2-medium
"""

import argparse
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dfastllm.engine.diffusion_sampler import (
    diffusion_generate,
    DiffusionSampler,
    DiffusionSamplerConfig,
    is_diffusion_model,
)


def test_diffusion_generate(
    model_name: str = "gpt2",
    prompt: str = "The future of AI is",
    gen_length: int = 16,
    steps: int = 8,
    block_length: int = 8,
    temperature: float = 0.0,
    verbose: bool = True,
):
    """Test diffusion generation with a real model.
    
    Args:
        model_name: HuggingFace model name
        prompt: Input prompt
        gen_length: Tokens to generate
        steps: Diffusion steps
        block_length: Block size for semi-AR generation
        temperature: Sampling temperature
        verbose: Print progress
    """
    print("=" * 60)
    print("dfastllm Diffusion Sampler Test")
    print("=" * 60)
    
    # Check if diffusion model
    is_diffusion = is_diffusion_model(model_name)
    print(f"\nModel: {model_name}")
    print(f"Is diffusion model: {is_diffusion}")
    
    # Load model and tokenizer
    print(f"\n[1/5] Loading model and tokenizer...")
    start = time.time()
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()
    
    print(f"      Model loaded in {time.time() - start:.2f}s")
    print(f"      Vocab size: {model.config.vocab_size:,}")
    print(f"      Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup mask token
    # For diffusion models (LLaDA), use their native mask token
    # For AR models (GPT-2, TinyLlama), use a special strategy
    if hasattr(tokenizer, 'mask_token_id') and tokenizer.mask_token_id:
        mask_id = tokenizer.mask_token_id
        print(f"      Mask token ID: {mask_id} (from tokenizer)")
    elif hasattr(tokenizer, 'unk_token_id') and tokenizer.unk_token_id:
        # Use UNK token as mask (better than arbitrary high ID)
        mask_id = tokenizer.unk_token_id
        print(f"      Mask token ID: {mask_id} (using UNK token)")
    else:
        # Fallback: use a token ID that's unlikely to be predicted
        # For LLaMA-based models, use 0 (usually <unk> or padding)
        mask_id = 0
        print(f"      Mask token ID: {mask_id} (fallback)")
    
    # Tokenize prompt
    print(f"\n[2/5] Tokenizing prompt...")
    inputs = tokenizer(prompt, return_tensors="pt")
    prompt_ids = inputs.input_ids
    prompt_length = prompt_ids.shape[1]
    print(f"      Prompt: '{prompt}'")
    print(f"      Prompt tokens: {prompt_ids.tolist()}")
    print(f"      Prompt length: {prompt_length}")
    
    # Adjust parameters for divisibility
    if gen_length % block_length != 0:
        block_length = gen_length
    num_blocks = gen_length // block_length
    if steps % num_blocks != 0:
        steps = max(num_blocks, (steps // num_blocks) * num_blocks)
    
    print(f"\n[3/5] Generation parameters:")
    print(f"      gen_length: {gen_length}")
    print(f"      steps: {steps}")
    print(f"      block_length: {block_length}")
    print(f"      temperature: {temperature}")
    
    # Run diffusion generation
    print(f"\n[4/5] Running diffusion generation...")
    start = time.time()
    
    with torch.no_grad():
        output_ids = diffusion_generate(
            model=model,
            prompt=prompt_ids,
            steps=steps,
            gen_length=gen_length,
            block_length=block_length,
            temperature=temperature,
            mask_id=mask_id,
            use_float32_gumbel=True,  # Faster on CPU
            enable_early_stopping=True,
        )
    
    gen_time = time.time() - start
    print(f"      Generation completed in {gen_time:.2f}s")
    print(f"      Tokens/sec: {gen_length / gen_time:.2f}")
    
    # Decode output
    print(f"\n[5/5] Decoding output...")
    generated_ids = output_ids[0, prompt_length:].tolist()
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    print(f"\n{'=' * 60}")
    print("RESULTS")
    print("=" * 60)
    print(f"\nPrompt: '{prompt}'")
    print(f"\nGenerated tokens: {generated_ids}")
    print(f"\nGenerated text: '{generated_text}'")
    print(f"\nFull output: '{full_text}'")
    
    # Verify no mask tokens remain
    num_masks = (output_ids == mask_id).sum().item()
    print(f"\nMask tokens remaining: {num_masks}")
    
    if num_masks == 0:
        print("\n✅ SUCCESS: All mask tokens were replaced!")
    else:
        print(f"\n⚠️  WARNING: {num_masks} mask tokens remain")
    
    return output_ids, generated_text


def test_diffusion_sampler_class(model_name: str = "gpt2"):
    """Test the DiffusionSampler class interface."""
    print("\n" + "=" * 60)
    print("Testing DiffusionSampler Class")
    print("=" * 60)
    
    # Load model and tokenizer
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()
    
    # Setup mask token
    if not hasattr(tokenizer, 'mask_token_id') or not tokenizer.mask_token_id:
        tokenizer.mask_token_id = model.config.vocab_size - 1
    
    # Create config
    config = DiffusionSamplerConfig(
        steps=8,
        block_length=8,
        temperature=0.0,
        use_float32_gumbel=True,
        enable_early_stopping=True,
    )
    
    # Create sampler
    print("Creating DiffusionSampler...")
    sampler = DiffusionSampler(model, tokenizer, config)
    
    # Generate
    prompt = "Hello world"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    print(f"Generating from prompt: '{prompt}'")
    start = time.time()
    
    output_ids = sampler.generate(inputs.input_ids, max_new_tokens=16)
    
    gen_time = time.time() - start
    print(f"Generation time: {gen_time:.2f}s")
    
    # Decode
    generated_text = sampler.decode(output_ids[0], prompt_length=inputs.input_ids.shape[1])
    print(f"Generated: '{generated_text}'")
    
    print("\n✅ DiffusionSampler class works!")


def main():
    parser = argparse.ArgumentParser(description="Test dfastllm diffusion sampler locally")
    parser.add_argument("--model", type=str, default="gpt2", help="Model name")
    parser.add_argument("--prompt", type=str, default="The future of AI is", help="Input prompt")
    parser.add_argument("--gen-length", type=int, default=16, help="Tokens to generate")
    parser.add_argument("--steps", type=int, default=8, help="Diffusion steps")
    parser.add_argument("--block-length", type=int, default=8, help="Block length")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature")
    parser.add_argument("--test-class", action="store_true", help="Also test DiffusionSampler class")
    
    args = parser.parse_args()
    
    # Run main test
    test_diffusion_generate(
        model_name=args.model,
        prompt=args.prompt,
        gen_length=args.gen_length,
        steps=args.steps,
        block_length=args.block_length,
        temperature=args.temperature,
    )
    
    # Optionally test class interface
    if args.test_class:
        test_diffusion_sampler_class(args.model)
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()


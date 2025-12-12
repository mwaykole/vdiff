#!/usr/bin/env python3
"""Minimal Diffusion LLM Demo.

This script demonstrates the core diffusion generation algorithm
without the full serving infrastructure. Useful for understanding
how vdiff and APD work.

Usage:
    python examples/diffusion_demo.py --model gpt2
    python examples/diffusion_demo.py --model GSAI-ML/LLaDA-8B-Instruct --use-apd
"""

import argparse
import time
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Apply Gumbel noise for sampling."""
    if temperature == 0:
        return logits
    
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64).clamp(min=1e-10)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


@torch.no_grad()
def diffusion_generate(
    model,
    prompt_ids: torch.Tensor,
    gen_length: int,
    steps: int,
    mask_id: int,
    temperature: float = 0.0,
    verbose: bool = True,
) -> torch.Tensor:
    """Simple diffusion generation (LLaDA-style).
    
    Args:
        model: Language model
        prompt_ids: Prompt token IDs
        gen_length: Tokens to generate
        steps: Diffusion steps
        mask_id: Mask token ID
        temperature: Sampling temperature
        verbose: Print progress
    
    Returns:
        Generated token IDs
    """
    device = next(model.parameters()).device
    batch_size = prompt_ids.shape[0]
    prompt_length = prompt_ids.shape[1]
    
    # Initialize: [prompt | MASK × gen_length]
    x = torch.full(
        (batch_size, prompt_length + gen_length),
        mask_id,
        dtype=torch.long,
        device=device
    )
    x[:, :prompt_length] = prompt_ids
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Diffusion Generation: {gen_length} tokens in {steps} steps")
        print(f"{'='*60}")
    
    # Calculate tokens to unmask per step
    tokens_per_step = gen_length // steps
    remainder = gen_length % steps
    
    for step in range(steps):
        # How many to unmask this step
        k = tokens_per_step + (1 if step < remainder else 0)
        
        # Forward pass
        outputs = model(x)
        logits = outputs.logits
        
        # Sample with Gumbel noise
        logits_noisy = add_gumbel_noise(logits, temperature)
        predictions = logits_noisy.argmax(dim=-1)
        
        # Compute confidence
        probs = F.softmax(logits.float(), dim=-1)
        confidence = probs.max(dim=-1).values
        
        # Only consider masked positions
        mask_positions = (x == mask_id)
        confidence = torch.where(mask_positions, confidence, torch.tensor(float('-inf'), device=device))
        
        # Select top-k
        for b in range(batch_size):
            num_masked = mask_positions[b].sum().item()
            k_actual = min(k, num_masked)
            
            if k_actual > 0:
                _, top_indices = torch.topk(confidence[b], k=k_actual)
                x[b, top_indices] = predictions[b, top_indices]
        
        if verbose:
            remaining = (x == mask_id).sum().item()
            print(f"Step {step+1}/{steps}: Unmasked {k} tokens, {remaining} remaining")
    
    return x


@torch.no_grad()
def apd_generate(
    model,
    prompt_ids: torch.Tensor,
    gen_length: int,
    steps: int,
    mask_id: int,
    max_parallel: int = 8,
    threshold: float = 0.3,
    temperature: float = 0.0,
    verbose: bool = True,
) -> torch.Tensor:
    """APD (Adaptive Parallel Decoding) generation.
    
    Args:
        model: Language model
        prompt_ids: Prompt token IDs
        gen_length: Tokens to generate
        steps: Max diffusion steps
        mask_id: Mask token ID
        max_parallel: Max tokens to try per step
        threshold: Acceptance threshold
        temperature: Sampling temperature
        verbose: Print progress
    
    Returns:
        Generated token IDs
    """
    device = next(model.parameters()).device
    batch_size = prompt_ids.shape[0]
    prompt_length = prompt_ids.shape[1]
    
    # Initialize
    x = torch.full(
        (batch_size, prompt_length + gen_length),
        mask_id,
        dtype=torch.long,
        device=device
    )
    x[:, :prompt_length] = prompt_ids
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"APD Generation: {gen_length} tokens, max_parallel={max_parallel}")
        print(f"{'='*60}")
    
    total_accepted = 0
    
    for step in range(steps):
        mask_positions = (x == mask_id)
        num_masked = mask_positions.sum().item()
        
        if num_masked == 0:
            if verbose:
                print(f"Step {step+1}: All tokens revealed! (early stop)")
            break
        
        # Forward pass
        outputs = model(x)
        logits = outputs.logits
        
        # Sample and compute confidence
        logits_noisy = add_gumbel_noise(logits, temperature)
        predictions = logits_noisy.argmax(dim=-1)
        probs = F.softmax(logits.float(), dim=-1)
        confidence = probs.max(dim=-1).values
        
        # Mask non-candidates
        confidence = torch.where(mask_positions, confidence, torch.tensor(float('-inf'), device=device))
        
        # Adaptive: try up to max_parallel
        remaining_steps = steps - step
        target = max(1, num_masked // remaining_steps)
        k = min(max_parallel, target, num_masked)
        
        accepted = 0
        for b in range(batch_size):
            _, top_indices = torch.topk(confidence[b], k=k)
            
            # Accept based on confidence threshold
            for idx in top_indices:
                if confidence[b, idx] > threshold:
                    x[b, idx] = predictions[b, idx]
                    accepted += 1
        
        total_accepted += accepted
        
        if verbose:
            remaining = (x == mask_id).sum().item()
            print(f"Step {step+1}/{steps}: Tried {k}, accepted {accepted}, {remaining} remaining")
    
    if verbose:
        print(f"Total accepted: {total_accepted}")
    
    return x


def main():
    parser = argparse.ArgumentParser(description="Diffusion LLM Demo")
    parser.add_argument("--model", type=str, default="gpt2", help="Model name")
    parser.add_argument("--prompt", type=str, default="Hello, how are you?", help="Prompt")
    parser.add_argument("--gen-length", type=int, default=20, help="Tokens to generate")
    parser.add_argument("--steps", type=int, default=10, help="Diffusion steps")
    parser.add_argument("--use-apd", action="store_true", help="Use APD instead of standard diffusion")
    parser.add_argument("--max-parallel", type=int, default=4, help="APD max parallel tokens")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/cuda)")
    args = parser.parse_args()
    
    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Loading model: {args.model}")
    print(f"Device: {device}")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()
    
    # Get mask token ID (use pad_token_id as fallback for non-diffusion models)
    if hasattr(tokenizer, 'mask_token_id') and tokenizer.mask_token_id is not None:
        mask_id = tokenizer.mask_token_id
    else:
        mask_id = tokenizer.pad_token_id or 0
        print(f"Note: Using pad_token_id ({mask_id}) as mask (model may not be a diffusion LLM)")
    
    # Tokenize prompt
    prompt_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(device)
    print(f"\nPrompt: {args.prompt}")
    print(f"Prompt tokens: {prompt_ids.shape[1]}")
    
    # Generate
    start_time = time.time()
    
    if args.use_apd:
        output_ids = apd_generate(
            model=model,
            prompt_ids=prompt_ids,
            gen_length=args.gen_length,
            steps=args.steps,
            mask_id=mask_id,
            max_parallel=args.max_parallel,
            temperature=args.temperature,
        )
    else:
        output_ids = diffusion_generate(
            model=model,
            prompt_ids=prompt_ids,
            gen_length=args.gen_length,
            steps=args.steps,
            mask_id=mask_id,
            temperature=args.temperature,
        )
    
    elapsed = time.time() - start_time
    
    # Decode
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    generated_text = output_text[len(args.prompt):].strip()
    
    print(f"\n{'='*60}")
    print("RESULT")
    print(f"{'='*60}")
    print(f"Full output: {output_text}")
    print(f"Generated: {generated_text}")
    print(f"Time: {elapsed*1000:.1f}ms")
    print(f"Speed: {args.gen_length/elapsed:.1f} tokens/sec")


if __name__ == "__main__":
    main()

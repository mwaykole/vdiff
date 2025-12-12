"""Diffusion Sampler for LLaDA and other Masked Diffusion LLMs.

Implementation based on the LLaDA generation algorithm.
Reference: https://github.com/ML-GSAI/LLaDA

Key concepts:
- Masked Diffusion: Start with [MASK] tokens, iteratively unmask
- Confidence-based remasking: Unmask high-confidence tokens first
- Semi-autoregressive: Block-by-block generation for efficiency
"""

from typing import Optional, Tuple, Literal
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn.functional as F
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")


@dataclass
class DiffusionSamplerConfig:
    """Configuration for diffusion sampling."""
    steps: int = 128
    gen_length: int = 128
    block_length: int = 32
    temperature: float = 0.0
    cfg_scale: float = 0.0
    remasking: Literal["low_confidence", "random"] = "low_confidence"
    mask_id: int = 126336  # LLaDA's mask token ID


def add_gumbel_noise(logits: "torch.Tensor", temperature: float) -> "torch.Tensor":
    """Apply Gumbel noise for sampling categorical distributions.
    
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves 
    perplexity score but reduces generation quality. Thus, we use float64.
    
    Args:
        logits: Model output logits.
        temperature: Sampling temperature (0 = greedy).
    
    Returns:
        Logits with Gumbel noise applied.
    """
    if temperature == 0:
        return logits
    
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(
    mask_index: "torch.Tensor",
    steps: int
) -> "torch.Tensor":
    """Compute number of tokens to unmask at each step.
    
    LLaDA uses a linear noise schedule, so the expected number of tokens
    transitioned at each step should be consistent.
    
    Args:
        mask_index: Boolean tensor indicating masked positions.
        steps: Number of sampling steps.
    
    Returns:
        Tensor of shape (batch_size, steps) with token counts per step.
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)
    
    base = mask_num // steps
    remainder = mask_num % steps
    
    num_transfer_tokens = torch.zeros(
        mask_num.size(0), steps, 
        device=mask_index.device, 
        dtype=torch.int64
    ) + base
    
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1
    
    return num_transfer_tokens


@torch.no_grad()
def diffusion_generate(
    model,
    prompt: "torch.Tensor",
    attention_mask: Optional["torch.Tensor"] = None,
    steps: int = 128,
    gen_length: int = 128,
    block_length: int = 32,
    temperature: float = 0.0,
    cfg_scale: float = 0.0,
    remasking: str = "low_confidence",
    mask_id: int = 126336,
) -> "torch.Tensor":
    """Generate text using masked diffusion.
    
    This implements the LLaDA generation algorithm which:
    1. Starts with all [MASK] tokens in the response
    2. Iteratively predicts and unmasks tokens based on confidence
    3. Supports block-based semi-autoregressive generation
    
    Args:
        model: The diffusion LLM (must have .forward() returning logits).
        prompt: Input token IDs of shape (batch_size, prompt_length).
        attention_mask: Optional attention mask.
        steps: Number of sampling steps (less than or equal to gen_length).
        gen_length: Length of generated response.
        block_length: Block size for semi-autoregressive generation.
        temperature: Gumbel noise temperature (0 = greedy).
        cfg_scale: Classifier-free guidance scale (0 = disabled).
        remasking: Strategy - "low_confidence" or "random".
        mask_id: Token ID for [MASK] token.
    
    Returns:
        Generated token IDs of shape (batch_size, prompt_length + gen_length).
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch required for diffusion generation")
    
    device = next(model.parameters()).device
    batch_size = prompt.shape[0]
    prompt_length = prompt.shape[1]
    
    # Initialize with masked tokens for response
    x = torch.full(
        (batch_size, prompt_length + gen_length),
        mask_id,
        dtype=torch.long,
        device=device
    )
    x[:, :prompt_length] = prompt.clone()
    
    # Extend attention mask if provided
    if attention_mask is not None:
        attention_mask = torch.cat([
            attention_mask,
            torch.ones((batch_size, gen_length), dtype=attention_mask.dtype, device=device)
        ], dim=-1)
    
    prompt_index = (x != mask_id)
    
    # Semi-autoregressive: process in blocks
    assert gen_length % block_length == 0, "gen_length must be divisible by block_length"
    num_blocks = gen_length // block_length
    
    assert steps % num_blocks == 0, "steps must be divisible by num_blocks"
    steps_per_block = steps // num_blocks
    
    for block_idx in range(num_blocks):
        block_start = prompt_length + block_idx * block_length
        block_end = prompt_length + (block_idx + 1) * block_length
        
        block_mask_index = (x[:, block_start:block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)
        
        for step in range(steps_per_block):
            mask_index = (x == mask_id)
            
            # Classifier-free guidance
            if cfg_scale > 0.0:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                
                if attention_mask is not None:
                    attention_mask_ = torch.cat([attention_mask, attention_mask], dim=0)
                else:
                    attention_mask_ = None
                
                outputs = model(x_, attention_mask=attention_mask_)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                outputs = model(x, attention_mask=attention_mask)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            
            # Apply Gumbel noise and sample
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)
            
            # Compute confidence for remasking
            if remasking == "low_confidence":
                p = F.softmax(logits.float(), dim=-1)
                x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
            elif remasking == "random":
                x0_p = torch.rand(x0.shape, device=device)
            else:
                raise ValueError(f"Unknown remasking strategy: {remasking}")
            
            # Don't consider positions after current block
            x0_p[:, block_end:] = float('-inf')
            
            # Only update masked positions
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, float('-inf'))
            
            # Select top-k confident tokens to unmask
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=device)
            for b in range(batch_size):
                k = num_transfer_tokens[b, step].item()
                if k > 0:
                    _, select_index = torch.topk(confidence[b], k=k)
                    transfer_index[b, select_index] = True
            
            x[transfer_index] = x0[transfer_index]
    
    return x


class DiffusionSampler:
    """High-level diffusion sampler for serving.
    
    Wraps the diffusion_generate function with configuration management.
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        config: Optional[DiffusionSamplerConfig] = None,
    ):
        """Initialize the diffusion sampler.
        
        Args:
            model: The diffusion LLM.
            tokenizer: Tokenizer with mask token.
            config: Sampling configuration.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or DiffusionSamplerConfig()
        
        # Auto-detect mask token ID
        if hasattr(tokenizer, 'mask_token_id') and tokenizer.mask_token_id:
            self.config.mask_id = tokenizer.mask_token_id
        
        logger.info(f"DiffusionSampler initialized with mask_id={self.config.mask_id}")
    
    def generate(
        self,
        input_ids: "torch.Tensor",
        attention_mask: Optional["torch.Tensor"] = None,
        max_new_tokens: int = 128,
        steps: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> "torch.Tensor":
        """Generate text using diffusion.
        
        Args:
            input_ids: Input token IDs.
            attention_mask: Optional attention mask.
            max_new_tokens: Maximum tokens to generate.
            steps: Override default steps.
            temperature: Override default temperature.
        
        Returns:
            Generated token IDs including prompt.
        """
        steps = steps or self.config.steps
        temperature = temperature if temperature is not None else self.config.temperature
        
        # Adjust block_length if needed
        block_length = min(self.config.block_length, max_new_tokens)
        if max_new_tokens % block_length != 0:
            block_length = max_new_tokens
        
        # Adjust steps to be divisible by num_blocks
        num_blocks = max_new_tokens // block_length
        if steps % num_blocks != 0:
            steps = (steps // num_blocks + 1) * num_blocks
        
        return diffusion_generate(
            model=self.model,
            prompt=input_ids,
            attention_mask=attention_mask,
            steps=steps,
            gen_length=max_new_tokens,
            block_length=block_length,
            temperature=temperature,
            cfg_scale=self.config.cfg_scale,
            remasking=self.config.remasking,
            mask_id=self.config.mask_id,
        )
    
    def decode(
        self,
        output_ids: "torch.Tensor",
        prompt_length: int,
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode generated tokens to text.
        
        Args:
            output_ids: Generated token IDs.
            prompt_length: Length of prompt to skip.
            skip_special_tokens: Whether to skip special tokens.
        
        Returns:
            Decoded text.
        """
        return self.tokenizer.decode(
            output_ids[prompt_length:],
            skip_special_tokens=skip_special_tokens
        )


def is_diffusion_model(model_name: str) -> bool:
    """Check if a model is a diffusion LLM based on name."""
    diffusion_models = [
        "llada", "LLaDA",
        "dream", "Dream",
        "mdlm", "MDLM",
        "smdm", "SMDM",
    ]
    return any(m.lower() in model_name.lower() for m in diffusion_models)

"""Diffusion Sampler for LLaDA and other Masked Diffusion LLMs.

Implementation based on the LLaDA generation algorithm.
Reference: https://github.com/ML-GSAI/LLaDA

Key concepts:
- Masked Diffusion: Start with [MASK] tokens, iteratively unmask
- Confidence-based remasking: Unmask high-confidence tokens first
- Semi-autoregressive: Block-by-block generation for efficiency

Optimizations:
- Vectorized top-k selection (no Python loops over batch)
- Memory-efficient tensor reuse
- Optional float32 Gumbel noise for speed
- Early stopping when fully unmasked
- torch.compile support
"""

from typing import Optional, Tuple, Literal
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")

@dataclass
class DiffusionSamplerConfig:
    """Configuration for diffusion sampling.
    
    Includes optimization options for production deployment:
    - Mixed precision (BF16/FP16) for faster forward passes
    - Adaptive steps for early stopping based on confidence
    - Attention caching for reusing stable attention patterns
    """
    steps: int = 128
    gen_length: int = 128
    block_length: int = 32
    temperature: float = 0.0
    cfg_scale: float = 0.0
    remasking: Literal["low_confidence", "random"] = "low_confidence"
    mask_id: int = 126336  # LLaDA's mask token ID
    
    # Optimization options
    use_float32_gumbel: bool = False  # Use float32 for speed (slight quality tradeoff)
    enable_early_stopping: bool = True  # Stop when all tokens unmasked
    use_mixed_precision: bool = True  # Use BF16/FP16 for forward pass
    use_adaptive_steps: bool = True  # Dynamically reduce steps based on confidence
    confidence_threshold: float = 0.95  # Threshold for adaptive early stopping
    enable_attention_cache: bool = False  # Cache attention maps (experimental)
    
    # Mixture of Recursions (MoR) options
    enable_mor: bool = True  # Enable MoR adaptive compute
    mor_min_recursions: int = 1  # Minimum recursions for easy tokens
    mor_max_recursions: int = 4  # Maximum recursions for hard tokens
    mor_confidence_high: float = 0.9  # Above this = skip extra recursions
    mor_confidence_low: float = 0.5  # Below this = apply max recursions

@torch.no_grad()
def add_gumbel_noise(
    logits: "torch.Tensor",
    temperature: float,
    use_float32: bool = False
) -> "torch.Tensor":
    """Apply Gumbel noise for sampling categorical distributions.
    
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves 
    perplexity score but reduces generation quality. Thus, we use float64 by default.
    
    Args:
        logits: Model output logits.
        temperature: Sampling temperature (0 = greedy).
        use_float32: Use float32 for speed (slight quality tradeoff).
    
    Returns:
        Logits with Gumbel noise applied.
    """
    if temperature == 0:
        return logits
    
    # Use float32 for speed if requested, otherwise float64 for quality
    dtype = torch.float32 if use_float32 else torch.float64
    logits = logits.to(dtype)
    
    # Clamp noise to avoid log(0)
    noise = torch.rand_like(logits, dtype=dtype).clamp_(min=1e-10)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise

@torch.no_grad()
def get_num_transfer_tokens(
    mask_index: "torch.Tensor",
    steps: int
) -> "torch.Tensor":
    """Compute number of tokens to unmask at each step (vectorized).
    
    LLaDA uses a linear noise schedule, so the expected number of tokens
    transitioned at each step should be consistent.
    
    Args:
        mask_index: Boolean tensor indicating masked positions.
        steps: Number of sampling steps.
    
    Returns:
        Tensor of shape (batch_size, steps) with token counts per step.
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)  # (batch_size, 1)
    
    base = mask_num // steps
    remainder = mask_num % steps
    
    # Vectorized: create step indices and compare with remainder
    step_indices = torch.arange(steps, device=mask_index.device).unsqueeze(0)  # (1, steps)
    num_transfer_tokens = base + (step_indices < remainder).long()
    
    return num_transfer_tokens

@torch.no_grad()
def _vectorized_topk_unmask(
    confidence: "torch.Tensor",
    num_tokens: "torch.Tensor",
    step: int,
) -> "torch.Tensor":
    """Vectorized top-k selection across batch (no Python loops).
    
    Args:
        confidence: Confidence scores (batch_size, seq_len).
        num_tokens: Tokens to unmask per batch item (batch_size, steps).
        step: Current step index.
    
    Returns:
        Boolean mask of positions to unmask.
    """
    batch_size, seq_len = confidence.shape
    device = confidence.device
    
    # Get max k across batch for uniform topk
    k_per_batch = num_tokens[:, step]  # (batch_size,)
    max_k = k_per_batch.max().item()
    
    if max_k == 0:
        return torch.zeros_like(confidence, dtype=torch.bool)
    
    # Get top-max_k indices for all batches
    _, top_indices = torch.topk(confidence, k=min(max_k, seq_len), dim=-1)  # (batch_size, max_k)
    
    # Create mask: only keep indices where position < k_per_batch[b]
    position_indices = torch.arange(max_k, device=device).unsqueeze(0)  # (1, max_k)
    valid_mask = position_indices < k_per_batch.unsqueeze(1)  # (batch_size, max_k)
    
    # Scatter to create transfer mask
    transfer_index = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    
    # Only set True for valid indices
    batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand_as(top_indices)
    valid_batch_indices = batch_indices[valid_mask]
    valid_top_indices = top_indices[valid_mask]
    
    transfer_index[valid_batch_indices, valid_top_indices] = True
    
    return transfer_index

@torch.no_grad()
def _apply_mor_refinement(
    model,
    x: "torch.Tensor",
    logits: "torch.Tensor",
    mask_index: "torch.Tensor",
    attention_mask: Optional["torch.Tensor"],
    min_recursions: int = 1,
    max_recursions: int = 4,
    confidence_high: float = 0.9,
    confidence_low: float = 0.5,
    use_amp: bool = False,
    amp_dtype: "torch.dtype" = None,
) -> "torch.Tensor":
    """Apply Mixture of Recursions refinement to logits.
    
    MoR allocates variable compute to tokens based on difficulty:
    - High confidence tokens: Skip refinement (already good)
    - Medium confidence: Light refinement
    - Low confidence tokens: Maximum refinement passes
    
    This implements Option 2 (inference-level MoR) which works with
    existing models without retraining.
    
    Args:
        model: The diffusion model.
        x: Current token sequence.
        logits: Initial logits from model.
        mask_index: Boolean mask of positions to consider.
        attention_mask: Optional attention mask.
        min_recursions: Minimum recursion depth.
        max_recursions: Maximum recursion depth.
        confidence_high: Above this = skip refinement.
        confidence_low: Below this = max refinement.
        use_amp: Whether to use automatic mixed precision.
        amp_dtype: Data type for AMP.
    
    Returns:
        Refined logits with adaptive compute applied.
    """
    if max_recursions <= 1:
        return logits
    
    device = logits.device
    batch_size, seq_len, vocab_size = logits.shape
    
    # Compute confidence for each masked position
    probs = F.softmax(logits.float(), dim=-1)
    confidence = probs.max(dim=-1).values  # (batch, seq_len)
    
    # Identify tokens by difficulty level
    # High confidence: no extra refinement needed
    high_conf_mask = mask_index & (confidence >= confidence_high)
    # Low confidence: needs maximum refinement
    low_conf_mask = mask_index & (confidence < confidence_low)
    # Medium: linear interpolation
    medium_mask = mask_index & ~high_conf_mask & ~low_conf_mask
    
    # Count how many tokens need refinement
    needs_refinement = low_conf_mask | medium_mask
    num_needing_refinement = needs_refinement.sum().item()
    
    if num_needing_refinement == 0:
        # All tokens are confident, skip refinement
        return logits
    
    # Calculate recursion depths
    # Map confidence to recursion count (inverse relationship)
    # confidence_low -> max_recursions, confidence_high -> min_recursions
    confidence_range = max(confidence_high - confidence_low, 1e-6)
    normalized_conf = (confidence - confidence_low) / confidence_range
    normalized_conf = normalized_conf.clamp(0, 1)
    
    # Inverse: low confidence = high recursions
    recursion_depth = max_recursions - (max_recursions - min_recursions) * normalized_conf
    recursion_depth = recursion_depth.round().long()
    
    # Only consider masked positions
    recursion_depth = torch.where(mask_index, recursion_depth, torch.zeros_like(recursion_depth))
    
    # Get unique depths > 1 that need processing
    unique_depths = torch.unique(recursion_depth[needs_refinement])
    unique_depths = unique_depths[unique_depths > 1]
    
    if len(unique_depths) == 0:
        return logits
    
    refined_logits = logits.clone()
    
    # Process by depth level for efficiency (batch similar work)
    for depth in sorted(unique_depths.tolist(), reverse=True):
        # Find all positions needing at least this depth
        needs_this_depth = needs_refinement & (recursion_depth >= depth)
        
        if not needs_this_depth.any():
            continue
        
        # Additional forward pass
        if use_amp and amp_dtype is not None:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                outputs = model(x, attention_mask=attention_mask)
                new_logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            new_logits = new_logits.float()
        else:
            outputs = model(x, attention_mask=attention_mask)
            new_logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        
        # Blend with existing logits using weighted average
        # Weight decreases with each iteration for stability
        blend_weight = 1.0 / depth
        
        # Apply blending only to positions needing this refinement
        for b in range(batch_size):
            positions = needs_this_depth[b].nonzero(as_tuple=True)[0]
            for pos in positions:
                refined_logits[b, pos] = (
                    (1 - blend_weight) * refined_logits[b, pos] +
                    blend_weight * new_logits[b, pos]
                )
    
    return refined_logits

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
    use_float32_gumbel: bool = False,
    enable_early_stopping: bool = True,
    use_mixed_precision: bool = True,
    use_adaptive_steps: bool = False,
    confidence_threshold: float = 0.95,
    enable_mor: bool = False,
    mor_min_recursions: int = 1,
    mor_max_recursions: int = 4,
    mor_confidence_high: float = 0.9,
    mor_confidence_low: float = 0.5,
) -> "torch.Tensor":
    """Generate text using masked diffusion (optimized).
    
    This implements the LLaDA generation algorithm which:
    1. Starts with all [MASK] tokens in the response
    2. Iteratively predicts and unmasks tokens based on confidence
    3. Supports block-based semi-autoregressive generation
    
    Optimizations:
    - Vectorized top-k selection (no Python batch loops)
    - Early stopping when all tokens are unmasked
    - Optional float32 Gumbel noise for speed
    - Pre-allocated tensors where possible
    - Mixed precision (BF16/FP16) for faster forward passes
    - Adaptive step reduction based on confidence
    
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
        use_float32_gumbel: Use float32 Gumbel noise for speed.
        enable_early_stopping: Stop early when all tokens unmasked.
        use_mixed_precision: Use BF16/FP16 for forward pass.
        use_adaptive_steps: Enable adaptive step reduction.
        confidence_threshold: Threshold for adaptive early stopping.
    
    Returns:
        Generated token IDs of shape (batch_size, prompt_length + gen_length).
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch required for diffusion generation")
    
    device = next(model.parameters()).device
    batch_size = prompt.shape[0]
    prompt_length = prompt.shape[1]
    total_length = prompt_length + gen_length
    
    # Initialize with masked tokens for response
    x = torch.full(
        (batch_size, total_length),
        mask_id,
        dtype=torch.long,
        device=device
    )
    x[:, :prompt_length] = prompt
    
    # Extend attention mask if provided
    if attention_mask is not None:
        attention_mask = torch.cat([
            attention_mask,
            torch.ones((batch_size, gen_length), dtype=attention_mask.dtype, device=device)
        ], dim=-1)
    
    # Pre-compute prompt index for CFG (immutable after init)
    prompt_index = torch.zeros(batch_size, total_length, dtype=torch.bool, device=device)
    prompt_index[:, :prompt_length] = True
    
    # Validate block structure
    if gen_length % block_length != 0:
        raise ValueError(
            f"gen_length ({gen_length}) must be divisible by block_length ({block_length})"
        )
    num_blocks = gen_length // block_length
    
    if steps % num_blocks != 0:
        raise ValueError(
            f"steps ({steps}) must be divisible by num_blocks ({num_blocks})"
        )
    steps_per_block = steps // num_blocks
    
    # Pre-allocate CFG tensors if needed
    if cfg_scale > 0.0:
        x_cfg = torch.empty((batch_size * 2, total_length), dtype=torch.long, device=device)
        if attention_mask is not None:
            attention_mask_cfg = attention_mask.repeat(2, 1)
        else:
            attention_mask_cfg = None
    
    # Pre-allocate confidence tensor for reuse
    neg_inf = torch.tensor(float('-inf'), device=device)
    
    # Setup mixed precision context
    use_amp = use_mixed_precision and device.type == "cuda"
    amp_dtype = torch.float16  # Default to float16
    if use_amp and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
    
    # Adaptive step tracking
    consecutive_high_confidence = 0
    total_masks = gen_length
    
    for block_idx in range(num_blocks):
        block_start = prompt_length + block_idx * block_length
        block_end = prompt_length + (block_idx + 1) * block_length
        
        # Compute transfer schedule for this block
        block_mask_index = (x[:, block_start:block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)
        
        for step in range(steps_per_block):
            # Check early stopping
            mask_index = (x == mask_id)
            masks_remaining = mask_index.sum().item()
            
            if enable_early_stopping and masks_remaining == 0:
                logger.debug(f"Early stop at block {block_idx}, step {step}: all unmasked")
                return x
            
            # Forward pass with optional mixed precision
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    if cfg_scale > 0.0:
                        x_cfg[:batch_size] = x
                        x_cfg[batch_size:] = x.clone()
                        x_cfg[batch_size:][prompt_index] = mask_id
                        outputs = model(x_cfg, attention_mask=attention_mask_cfg)
                    else:
                        outputs = model(x, attention_mask=attention_mask)
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                # Convert back to float32 for stability
                logits = logits.float()
            else:
                if cfg_scale > 0.0:
                    x_cfg[:batch_size] = x
                    x_cfg[batch_size:] = x.clone()
                    x_cfg[batch_size:][prompt_index] = mask_id
                    outputs = model(x_cfg, attention_mask=attention_mask_cfg)
                else:
                    outputs = model(x, attention_mask=attention_mask)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            
            # Apply CFG if enabled
            if cfg_scale > 0.0:
                logits, un_logits = logits[:batch_size], logits[batch_size:]
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            
            # MoR: Mixture of Recursions - Adaptive refinement for difficult tokens
            if enable_mor and mask_index.any():
                logits = _apply_mor_refinement(
                    model=model,
                    x=x,
                    logits=logits,
                    mask_index=mask_index,
                    attention_mask=attention_mask,
                    min_recursions=mor_min_recursions,
                    max_recursions=mor_max_recursions,
                    confidence_high=mor_confidence_high,
                    confidence_low=mor_confidence_low,
                    use_amp=use_amp,
                    amp_dtype=amp_dtype,
                )
            
            # Sample with Gumbel noise
            logits_with_noise = add_gumbel_noise(logits, temperature, use_float32_gumbel)
            x0 = torch.argmax(logits_with_noise, dim=-1)
            
            # Compute confidence for remasking
            if remasking == "low_confidence":
                p = F.softmax(logits.float(), dim=-1)
                x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
            elif remasking == "random":
                x0_p = torch.rand(x0.shape, device=device)
            else:
                raise ValueError(f"Unknown remasking strategy: {remasking}")
            
            # Adaptive step check based on confidence
            if use_adaptive_steps and remasking == "low_confidence":
                valid_confidence = x0_p[mask_index]
                if valid_confidence.numel() > 0:
                    avg_confidence = valid_confidence.mean().item()
                    unmasked_ratio = 1.0 - (masks_remaining / total_masks)
                    
                    # Check if we should stop early
                    if avg_confidence >= confidence_threshold:
                        consecutive_high_confidence += 1
                        if consecutive_high_confidence >= 2:  # Patience of 2
                            logger.debug(
                                f"Adaptive early stop at block {block_idx}, step {step}: "
                                f"confidence={avg_confidence:.3f}, unmasked={unmasked_ratio:.1%}"
                            )
                            # Unmask remaining high-confidence tokens before returning
                            remaining_mask = (x == mask_id)
                            if remaining_mask.any():
                                x = torch.where(remaining_mask, x0, x)
                            return x
                    else:
                        consecutive_high_confidence = 0
                    
                    # Also stop if mostly unmasked
                    if unmasked_ratio >= 0.95:
                        logger.debug(f"Adaptive stop: {unmasked_ratio:.1%} unmasked")
                        return x
            
            # Mask out: prompt, already unmasked, and future blocks
            x0_p = torch.where(mask_index, x0_p, neg_inf)
            x0_p[:, block_end:] = neg_inf
            
            # Only update masked positions with predictions
            x0 = torch.where(mask_index, x0, x)
            
            # Vectorized top-k selection (no Python loop!)
            transfer_index = _vectorized_topk_unmask(x0_p, num_transfer_tokens, step)
            
            # Apply updates
            x = torch.where(transfer_index, x0, x)
    
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
        
        logger.info(
            f"DiffusionSampler initialized: mask_id={self.config.mask_id}, "
            f"early_stop={self.config.enable_early_stopping}, "
            f"mixed_precision={self.config.use_mixed_precision}, "
            f"adaptive_steps={self.config.use_adaptive_steps}, "
            f"mor={self.config.enable_mor} (recursions={self.config.mor_min_recursions}-{self.config.mor_max_recursions})"
        )
    
    def generate(
        self,
        input_ids: "torch.Tensor",
        attention_mask: Optional["torch.Tensor"] = None,
        max_new_tokens: int = 128,
        steps: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> "torch.Tensor":
        """Generate text using diffusion (optimized).
        
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
            # Find largest divisor <= block_length
            for bl in range(block_length, 0, -1):
                if max_new_tokens % bl == 0:
                    block_length = bl
                    break
            else:
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
            use_float32_gumbel=self.config.use_float32_gumbel,
            enable_early_stopping=self.config.enable_early_stopping,
            use_mixed_precision=self.config.use_mixed_precision,
            use_adaptive_steps=self.config.use_adaptive_steps,
            confidence_threshold=self.config.confidence_threshold,
            enable_mor=self.config.enable_mor,
            mor_min_recursions=self.config.mor_min_recursions,
            mor_max_recursions=self.config.mor_max_recursions,
            mor_confidence_high=self.config.mor_confidence_high,
            mor_confidence_low=self.config.mor_confidence_low,
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

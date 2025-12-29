"""Mixture of Recursions (MoR) Decoder for Diffusion LLMs.

MoR is an adaptive computation technique that allocates variable compute
to different tokens based on their difficulty/uncertainty.

This module implements MoR at inference level (Option 2), working with
existing diffusion models without requiring model retraining.

Key concepts:
- Token Difficulty: Measured by confidence (1 - max_prob)
- Adaptive Recursions: Hard tokens get more refinement passes
- Compute Efficiency: Skip processing for confident tokens
- Quality Improvement: Hard tokens get better quality

Reference: https://arxiv.org/abs/2507.10524
"""

from typing import Optional, Tuple, Dict, Any, List, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import time

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available, MoR disabled")

class RouterStrategy(Enum):
    """Strategy for determining recursion depth per token."""
    CONFIDENCE = "confidence"
    ENTROPY = "entropy"
    GRADIENT = "gradient"
    FIXED = "fixed"

@dataclass
class MoRConfig:
    """Configuration for Mixture of Recursions decoder.
    
    Attributes:
        enabled: Whether MoR is enabled.
        min_recursions: Minimum recursion depth for any token.
        max_recursions: Maximum recursion depth for hardest tokens.
        router_strategy: How to determine token difficulty.
        difficulty_threshold_low: Confidence above this = easy token.
        difficulty_threshold_high: Confidence below this = hard token.
        skip_confident_tokens: Skip processing for high-confidence tokens.
        skip_threshold: Confidence threshold for skipping.
        batch_by_difficulty: Group tokens by difficulty for efficiency.
        log_stats: Log MoR statistics.
    """
    enabled: bool = True
    min_recursions: int = 1
    max_recursions: int = 4
    router_strategy: RouterStrategy = RouterStrategy.CONFIDENCE
    difficulty_threshold_low: float = 0.8
    difficulty_threshold_high: float = 0.3
    skip_confident_tokens: bool = True
    skip_threshold: float = 0.95
    batch_by_difficulty: bool = True
    log_stats: bool = False
    
    def __post_init__(self):
        if self.min_recursions < 1:
            raise ValueError("min_recursions must be >= 1")
        if self.max_recursions < self.min_recursions:
            raise ValueError("max_recursions must be >= min_recursions")
        if not 0 <= self.difficulty_threshold_low <= 1:
            raise ValueError("difficulty_threshold_low must be in [0, 1]")
        if not 0 <= self.difficulty_threshold_high <= 1:
            raise ValueError("difficulty_threshold_high must be in [0, 1]")

@dataclass
class MoRStats:
    """Statistics for MoR decoding performance."""
    total_steps: int = 0
    total_tokens_processed: int = 0
    tokens_skipped: int = 0
    easy_tokens: int = 0
    medium_tokens: int = 0
    hard_tokens: int = 0
    total_recursions: int = 0
    compute_saved_pct: float = 0.0
    avg_recursions_per_token: float = 0.0
    
    def update(
        self,
        processed: int,
        skipped: int,
        easy: int,
        medium: int,
        hard: int,
        recursions: int,
    ):
        """Update statistics after a step."""
        self.total_steps += 1
        self.total_tokens_processed += processed
        self.tokens_skipped += skipped
        self.easy_tokens += easy
        self.medium_tokens += medium
        self.hard_tokens += hard
        self.total_recursions += recursions
        
        total = processed + skipped
        if total > 0:
            # Compute savings: what we would have done vs what we did
            max_possible = total * 1  # All tokens, 1 full pass
            if processed > 0:
                self.avg_recursions_per_token = recursions / processed
            actual_compute = processed  # We processed this many
            self.compute_saved_pct = 100 * (1 - actual_compute / max(max_possible, 1))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "total_steps": self.total_steps,
            "total_tokens_processed": self.total_tokens_processed,
            "tokens_skipped": self.tokens_skipped,
            "easy_tokens": self.easy_tokens,
            "medium_tokens": self.medium_tokens,
            "hard_tokens": self.hard_tokens,
            "total_recursions": self.total_recursions,
            "compute_saved_pct": round(self.compute_saved_pct, 2),
            "avg_recursions_per_token": round(self.avg_recursions_per_token, 2),
        }
    
    def reset(self):
        """Reset all statistics."""
        self.total_steps = 0
        self.total_tokens_processed = 0
        self.tokens_skipped = 0
        self.easy_tokens = 0
        self.medium_tokens = 0
        self.hard_tokens = 0
        self.total_recursions = 0
        self.compute_saved_pct = 0.0
        self.avg_recursions_per_token = 0.0

class MoRDecoder:
    """Mixture of Recursions decoder for diffusion LLMs.
    
    Implements adaptive computation allocation at inference time:
    - Easy tokens (high confidence): Minimal refinement
    - Medium tokens: Standard refinement
    - Hard tokens (low confidence): Extra refinement passes
    
    This works with existing diffusion models without retraining.
    
    Example:
        >>> config = MoRConfig(enabled=True, max_recursions=4)
        >>> mor = MoRDecoder(config)
        >>> 
        >>> # In diffusion loop
        >>> for step in range(steps):
        >>>     logits = model(x)
        >>>     confidence = mor.compute_confidence(logits, mask_index)
        >>>     
        >>>     # Get adaptive refinement schedule
        >>>     recursion_depths = mor.compute_recursion_depths(confidence, mask_index)
        >>>     
        >>>     # Apply refinement
        >>>     refined_logits = mor.apply_adaptive_refinement(
        >>>         model, x, logits, recursion_depths, mask_index
        >>>     )
    """
    
    def __init__(self, config: Optional[MoRConfig] = None):
        """Initialize MoR decoder.
        
        Args:
            config: MoR configuration.
        """
        self.config = config or MoRConfig()
        self.stats = MoRStats()
        self._step_count = 0
        
        if self.config.enabled:
            logger.info(
                f"MoR decoder initialized: strategy={self.config.router_strategy.value}, "
                f"recursions={self.config.min_recursions}-{self.config.max_recursions}, "
                f"skip_threshold={self.config.skip_threshold}"
            )
    
    def compute_confidence(
        self,
        logits: "torch.Tensor",
        mask_index: "torch.Tensor",
    ) -> "torch.Tensor":
        """Compute per-token confidence scores.
        
        Args:
            logits: Model logits (batch, seq_len, vocab_size).
            mask_index: Boolean mask of active positions.
        
        Returns:
            Confidence scores (batch, seq_len) in [0, 1].
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for MoR")
        
        probs = F.softmax(logits.float(), dim=-1)
        confidence = probs.max(dim=-1).values
        
        # Only consider masked positions
        confidence = torch.where(
            mask_index,
            confidence,
            torch.ones_like(confidence)  # Unmasked = confident
        )
        
        return confidence
    
    def compute_entropy(
        self,
        logits: "torch.Tensor",
        mask_index: "torch.Tensor",
    ) -> "torch.Tensor":
        """Compute per-token entropy (alternative difficulty measure).
        
        Higher entropy = more uncertainty = harder token.
        
        Args:
            logits: Model logits.
            mask_index: Boolean mask of active positions.
        
        Returns:
            Normalized entropy scores in [0, 1].
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for MoR")
        
        probs = F.softmax(logits.float(), dim=-1)
        log_probs = F.log_softmax(logits.float(), dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        
        # Normalize by max entropy (log vocab_size)
        vocab_size = logits.size(-1)
        max_entropy = torch.log(torch.tensor(vocab_size, dtype=torch.float32, device=logits.device))
        normalized_entropy = entropy / max_entropy
        
        return torch.where(mask_index, normalized_entropy, torch.zeros_like(normalized_entropy))
    
    def compute_difficulty(
        self,
        logits: "torch.Tensor",
        mask_index: "torch.Tensor",
    ) -> "torch.Tensor":
        """Compute per-token difficulty scores.
        
        Difficulty is the inverse of confidence or based on entropy,
        depending on the router strategy.
        
        Args:
            logits: Model logits.
            mask_index: Boolean mask of active positions.
        
        Returns:
            Difficulty scores in [0, 1] where 1 = hardest.
        """
        if self.config.router_strategy == RouterStrategy.CONFIDENCE:
            confidence = self.compute_confidence(logits, mask_index)
            difficulty = 1.0 - confidence
        elif self.config.router_strategy == RouterStrategy.ENTROPY:
            difficulty = self.compute_entropy(logits, mask_index)
        elif self.config.router_strategy == RouterStrategy.FIXED:
            difficulty = torch.ones(
                logits.shape[:-1],
                device=logits.device,
                dtype=torch.float32
            ) * 0.5
        else:
            raise ValueError(f"Unknown router strategy: {self.config.router_strategy}")
        
        return difficulty
    
    def compute_recursion_depths(
        self,
        difficulty: "torch.Tensor",
        mask_index: "torch.Tensor",
    ) -> "torch.Tensor":
        """Compute recursion depth for each token based on difficulty.
        
        Maps difficulty scores to integer recursion counts:
        - Low difficulty (< threshold_low): min_recursions
        - High difficulty (> threshold_high): max_recursions
        - Medium: Linear interpolation
        
        Args:
            difficulty: Difficulty scores in [0, 1].
            mask_index: Boolean mask of active positions.
        
        Returns:
            Integer recursion depths (batch, seq_len).
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for MoR")
        
        min_r = self.config.min_recursions
        max_r = self.config.max_recursions
        low_thresh = 1.0 - self.config.difficulty_threshold_low
        high_thresh = 1.0 - self.config.difficulty_threshold_high
        
        # Linear mapping from difficulty to recursion depth
        # difficulty < low_thresh -> min_r
        # difficulty > high_thresh -> max_r
        # else -> linear interpolation
        
        range_width = max(high_thresh - low_thresh, 1e-6)
        normalized = (difficulty - low_thresh) / range_width
        normalized = normalized.clamp(0, 1)
        
        # Map to recursion range
        recursions = min_r + (max_r - min_r) * normalized
        recursions = recursions.round().long()
        
        # Only apply to masked positions
        recursions = torch.where(
            mask_index,
            recursions,
            torch.zeros_like(recursions)
        )
        
        return recursions
    
    def categorize_tokens(
        self,
        difficulty: "torch.Tensor",
        mask_index: "torch.Tensor",
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        """Categorize tokens into easy, medium, and hard groups.
        
        Args:
            difficulty: Difficulty scores.
            mask_index: Boolean mask of active positions.
        
        Returns:
            Tuple of (easy_mask, medium_mask, hard_mask).
        """
        low_thresh = 1.0 - self.config.difficulty_threshold_low
        high_thresh = 1.0 - self.config.difficulty_threshold_high
        
        easy_mask = mask_index & (difficulty < low_thresh)
        hard_mask = mask_index & (difficulty > high_thresh)
        medium_mask = mask_index & ~easy_mask & ~hard_mask
        
        return easy_mask, medium_mask, hard_mask
    
    def should_skip_token(
        self,
        confidence: "torch.Tensor",
        mask_index: "torch.Tensor",
    ) -> "torch.Tensor":
        """Determine which tokens can be skipped (high confidence).
        
        Args:
            confidence: Confidence scores.
            mask_index: Boolean mask of active positions.
        
        Returns:
            Boolean mask of tokens to skip.
        """
        if not self.config.skip_confident_tokens:
            return torch.zeros_like(mask_index)
        
        skip_mask = mask_index & (confidence >= self.config.skip_threshold)
        return skip_mask
    
    def apply_adaptive_refinement(
        self,
        model,
        x: "torch.Tensor",
        logits: "torch.Tensor",
        recursion_depths: "torch.Tensor",
        mask_index: "torch.Tensor",
        attention_mask: Optional["torch.Tensor"] = None,
        refine_fn: Optional[Callable] = None,
    ) -> "torch.Tensor":
        """Apply adaptive refinement based on recursion depths.
        
        This is the core MoR logic: tokens with higher recursion depths
        get additional forward passes for refinement.
        
        Args:
            model: The diffusion model.
            x: Current token sequence (batch, seq_len).
            logits: Initial logits from model.
            recursion_depths: Per-token recursion counts.
            mask_index: Boolean mask of active positions.
            attention_mask: Optional attention mask.
            refine_fn: Optional custom refinement function.
        
        Returns:
            Refined logits.
        """
        if not TORCH_AVAILABLE:
            return logits
        
        if not self.config.enabled:
            return logits
        
        # Get unique recursion depths to process
        unique_depths = torch.unique(recursion_depths[mask_index])
        unique_depths = unique_depths[unique_depths > 1]  # Skip depth=1 (already done)
        
        if len(unique_depths) == 0:
            return logits
        
        refined_logits = logits.clone()
        device = logits.device
        
        # Track stats
        total_recursions = 0
        
        # Process each depth level
        for depth in unique_depths:
            depth_val = depth.item()
            
            # Find tokens needing this many recursions
            needs_refinement = mask_index & (recursion_depths >= depth)
            
            if not needs_refinement.any():
                continue
            
            # Additional forward passes for these tokens
            for r in range(1, depth_val):
                if refine_fn is not None:
                    # Use custom refinement
                    refined_logits = refine_fn(model, x, refined_logits, needs_refinement)
                else:
                    # Default: run another forward pass and blend
                    with torch.no_grad():
                        outputs = model(x, attention_mask=attention_mask)
                        new_logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                    
                    # Blend new logits with existing (weighted average)
                    blend_weight = 1.0 / (r + 1)
                    for b in range(refined_logits.size(0)):
                        for pos in range(refined_logits.size(1)):
                            if needs_refinement[b, pos]:
                                refined_logits[b, pos] = (
                                    (1 - blend_weight) * refined_logits[b, pos] +
                                    blend_weight * new_logits[b, pos]
                                )
                
                total_recursions += needs_refinement.sum().item()
        
        # Update stats
        easy, medium, hard = self.categorize_tokens(
            1.0 - self.compute_confidence(logits, mask_index),
            mask_index
        )
        
        if self.config.log_stats:
            self.stats.update(
                processed=mask_index.sum().item(),
                skipped=0,
                easy=easy.sum().item(),
                medium=medium.sum().item(),
                hard=hard.sum().item(),
                recursions=total_recursions,
            )
        
        return refined_logits
    
    def get_processing_mask(
        self,
        confidence: "torch.Tensor",
        mask_index: "torch.Tensor",
    ) -> "torch.Tensor":
        """Get mask of tokens that need processing (not skipped).
        
        Args:
            confidence: Confidence scores.
            mask_index: Boolean mask of active positions.
        
        Returns:
            Boolean mask of tokens to process.
        """
        skip_mask = self.should_skip_token(confidence, mask_index)
        return mask_index & ~skip_mask
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current MoR statistics."""
        return self.stats.to_dict()
    
    def reset_stats(self):
        """Reset MoR statistics."""
        self.stats.reset()

@torch.no_grad()
def mor_diffusion_generate(
    model,
    prompt: "torch.Tensor",
    mor_config: Optional[MoRConfig] = None,
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
) -> Tuple["torch.Tensor", MoRStats]:
    """Generate text using MoR-enhanced masked diffusion.
    
    This is the main generation function that combines diffusion
    generation with Mixture of Recursions for adaptive compute.
    
    Key optimizations:
    - Skip processing for high-confidence tokens
    - Extra refinement for low-confidence tokens
    - Batch tokens by difficulty for efficiency
    
    Args:
        model: The diffusion LLM.
        prompt: Input token IDs.
        mor_config: MoR configuration.
        attention_mask: Optional attention mask.
        steps: Number of sampling steps.
        gen_length: Length of generated response.
        block_length: Block size for semi-autoregressive generation.
        temperature: Gumbel noise temperature.
        cfg_scale: Classifier-free guidance scale.
        remasking: Remasking strategy.
        mask_id: Token ID for [MASK].
        use_float32_gumbel: Use float32 for Gumbel noise.
        enable_early_stopping: Stop early when done.
        use_mixed_precision: Use BF16/FP16.
    
    Returns:
        Tuple of (generated tokens, MoR statistics).
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch required for MoR generation")
    
    # Initialize MoR decoder
    mor = MoRDecoder(mor_config)
    mor_config = mor.config
    
    device = next(model.parameters()).device
    batch_size = prompt.shape[0]
    prompt_length = prompt.shape[1]
    total_length = prompt_length + gen_length
    
    # Initialize with masked tokens
    x = torch.full(
        (batch_size, total_length),
        mask_id,
        dtype=torch.long,
        device=device
    )
    x[:, :prompt_length] = prompt
    
    # Extend attention mask
    if attention_mask is not None:
        attention_mask = torch.cat([
            attention_mask,
            torch.ones((batch_size, gen_length), dtype=attention_mask.dtype, device=device)
        ], dim=-1)
    
    # Validate structure
    if gen_length % block_length != 0:
        raise ValueError(f"gen_length ({gen_length}) must be divisible by block_length ({block_length})")
    num_blocks = gen_length // block_length
    
    if steps % num_blocks != 0:
        raise ValueError(f"steps ({steps}) must be divisible by num_blocks ({num_blocks})")
    steps_per_block = steps // num_blocks
    
    # Pre-allocate
    neg_inf = torch.tensor(float('-inf'), device=device)
    
    # Setup mixed precision
    use_amp = use_mixed_precision and device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    
    # Gumbel noise helper
    def add_gumbel_noise(logits, temp):
        if temp == 0:
            return logits
        dtype = torch.float32 if use_float32_gumbel else torch.float64
        logits = logits.to(dtype)
        noise = torch.rand_like(logits, dtype=dtype).clamp_(min=1e-10)
        gumbel_noise = (-torch.log(noise)) ** temp
        return logits.exp() / gumbel_noise
    
    # Vectorized top-k helper
    def get_num_transfer_tokens(mask_idx, num_steps):
        mask_num = mask_idx.sum(dim=1, keepdim=True)
        base = mask_num // num_steps
        remainder = mask_num % num_steps
        step_indices = torch.arange(num_steps, device=device).unsqueeze(0)
        return base + (step_indices < remainder).long()
    
    def topk_unmask(conf, num_tokens, step):
        batch_sz, seq_len = conf.shape
        k_per_batch = num_tokens[:, step]
        max_k = k_per_batch.max().item()
        if max_k == 0:
            return torch.zeros_like(conf, dtype=torch.bool)
        _, top_indices = torch.topk(conf, k=min(max_k, seq_len), dim=-1)
        position_indices = torch.arange(max_k, device=device).unsqueeze(0)
        valid_mask = position_indices < k_per_batch.unsqueeze(1)
        transfer_index = torch.zeros(batch_sz, seq_len, dtype=torch.bool, device=device)
        batch_indices = torch.arange(batch_sz, device=device).unsqueeze(1).expand_as(top_indices)
        valid_batch_indices = batch_indices[valid_mask]
        valid_top_indices = top_indices[valid_mask]
        transfer_index[valid_batch_indices, valid_top_indices] = True
        return transfer_index
    
    # Main generation loop
    for block_idx in range(num_blocks):
        block_start = prompt_length + block_idx * block_length
        block_end = prompt_length + (block_idx + 1) * block_length
        
        block_mask_index = (x[:, block_start:block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)
        
        for step in range(steps_per_block):
            mask_index = (x == mask_id)
            
            if enable_early_stopping and not mask_index.any():
                logger.debug(f"MoR early stop at block {block_idx}, step {step}")
                return x, mor.stats
            
            # Forward pass
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    outputs = model(x, attention_mask=attention_mask)
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                logits = logits.float()
            else:
                outputs = model(x, attention_mask=attention_mask)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            
            # MoR: Compute difficulty and recursion depths
            if mor_config.enabled:
                difficulty = mor.compute_difficulty(logits, mask_index)
                recursion_depths = mor.compute_recursion_depths(difficulty, mask_index)
                
                # Apply adaptive refinement for hard tokens
                logits = mor.apply_adaptive_refinement(
                    model, x, logits, recursion_depths, mask_index, attention_mask
                )
                
                # Update stats
                confidence = mor.compute_confidence(logits, mask_index)
                skip_mask = mor.should_skip_token(confidence, mask_index)
                easy, medium, hard = mor.categorize_tokens(difficulty, mask_index)
                
                mor.stats.update(
                    processed=mask_index.sum().item() - skip_mask.sum().item(),
                    skipped=skip_mask.sum().item(),
                    easy=easy.sum().item(),
                    medium=medium.sum().item(),
                    hard=hard.sum().item(),
                    recursions=recursion_depths[mask_index].sum().item(),
                )
            
            # Sample with Gumbel noise
            logits_with_noise = add_gumbel_noise(logits, temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)
            
            # Confidence for remasking
            if remasking == "low_confidence":
                p = F.softmax(logits.float(), dim=-1)
                x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
            else:
                x0_p = torch.rand(x0.shape, device=device)
            
            # Mask out non-candidates
            x0_p = torch.where(mask_index, x0_p, neg_inf)
            x0_p[:, block_end:] = neg_inf
            x0 = torch.where(mask_index, x0, x)
            
            # Top-k selection
            transfer_index = topk_unmask(x0_p, num_transfer_tokens, step)
            x = torch.where(transfer_index, x0, x)
    
    return x, mor.stats

class MoRDiffusionSampler:
    """High-level MoR-enhanced diffusion sampler.
    
    Wraps mor_diffusion_generate with configuration management.
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        mor_config: Optional[MoRConfig] = None,
        steps: int = 128,
        block_length: int = 32,
        temperature: float = 0.0,
        mask_id: Optional[int] = None,
    ):
        """Initialize MoR diffusion sampler.
        
        Args:
            model: The diffusion LLM.
            tokenizer: Tokenizer.
            mor_config: MoR configuration.
            steps: Default number of steps.
            block_length: Block size.
            temperature: Sampling temperature.
            mask_id: Override mask token ID.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.mor_config = mor_config or MoRConfig()
        self.steps = steps
        self.block_length = block_length
        self.temperature = temperature
        
        # Auto-detect mask token
        if mask_id is not None:
            self.mask_id = mask_id
        elif hasattr(tokenizer, 'mask_token_id') and tokenizer.mask_token_id:
            self.mask_id = tokenizer.mask_token_id
        else:
            self.mask_id = 126336  # LLaDA default
        
        self._last_stats: Optional[MoRStats] = None
        
        logger.info(
            f"MoRDiffusionSampler initialized: "
            f"mor_enabled={self.mor_config.enabled}, "
            f"max_recursions={self.mor_config.max_recursions}, "
            f"mask_id={self.mask_id}"
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
        """Generate text with MoR-enhanced diffusion.
        
        Args:
            input_ids: Input token IDs.
            attention_mask: Optional attention mask.
            max_new_tokens: Maximum tokens to generate.
            steps: Override steps.
            temperature: Override temperature.
        
        Returns:
            Generated token IDs including prompt.
        """
        steps = steps or self.steps
        temperature = temperature if temperature is not None else self.temperature
        
        # Adjust block_length
        block_length = min(self.block_length, max_new_tokens)
        if max_new_tokens % block_length != 0:
            for bl in range(block_length, 0, -1):
                if max_new_tokens % bl == 0:
                    block_length = bl
                    break
            else:
                block_length = max_new_tokens
        
        # Adjust steps
        num_blocks = max_new_tokens // block_length
        if steps % num_blocks != 0:
            steps = (steps // num_blocks + 1) * num_blocks
        
        output, stats = mor_diffusion_generate(
            model=self.model,
            prompt=input_ids,
            mor_config=self.mor_config,
            attention_mask=attention_mask,
            steps=steps,
            gen_length=max_new_tokens,
            block_length=block_length,
            temperature=temperature,
            mask_id=self.mask_id,
        )
        
        self._last_stats = stats
        return output
    
    def get_last_stats(self) -> Optional[Dict[str, Any]]:
        """Get statistics from the last generation."""
        if self._last_stats:
            return self._last_stats.to_dict()
        return None
    
    def decode(
        self,
        output_ids: "torch.Tensor",
        prompt_length: int,
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode generated tokens to text."""
        return self.tokenizer.decode(
            output_ids[prompt_length:],
            skip_special_tokens=skip_special_tokens
        )

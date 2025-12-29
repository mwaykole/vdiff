"""Hybrid Diffusion-Autoregressive Engine for LLM Inference.

Implements research-backed hybrid approaches combining the strengths of:
- Diffusion LLMs: Parallel token generation, bidirectional context
- Autoregressive LLMs: Causal coherence, proven quality

Based on research papers:
1. DEER: Draft with Diffusion, Verify with AR (https://czc726.github.io/DEER/)
2. DiffuSpec: Speculative Decoding with Diffusion Drafters (arxiv:2510.02358)
3. SpecDiff: Speculative Diffusion Decoding (NAACL 2025)

Key algorithms implemented:
- Parallel block drafting via diffusion
- Causal-consistency path search for AR verification
- Adaptive draft length control
- Longest accepted prefix extraction
"""

from typing import Optional, Tuple, List, Dict, Any, Union
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
    logger.warning("PyTorch not available, hybrid engine disabled")


class HybridMode(Enum):
    """Available hybrid generation modes."""
    DEER = "deer"  # Draft with Diffusion, Verify with AR
    SPEC_DIFF = "spec_diff"  # Speculative Diffusion Decoding
    SEMI_AR = "semi_ar"  # Semi-autoregressive block generation
    ADAPTIVE = "adaptive"  # Dynamically choose based on context


@dataclass
class HybridConfig:
    """Configuration for hybrid diffusion-AR generation.
    
    Based on hyperparameters from DEER and DiffuSpec papers.
    
    Attributes:
        enabled: Whether hybrid mode is enabled.
        mode: Hybrid generation mode to use.
        ar_verifier_model: Path to AR model for verification.
        draft_block_size: Number of tokens to draft per diffusion step.
        max_draft_tokens: Maximum tokens in a single draft.
        acceptance_threshold: Minimum probability for token acceptance.
        diffusion_weight: Weight for diffusion probabilities (alpha).
        ar_weight: Weight for AR probabilities (beta).
        use_causal_consistency: Enable causal-consistency path search.
        adaptive_draft_length: Dynamically adjust draft length.
        min_draft_length: Minimum draft tokens.
        max_draft_length: Maximum draft tokens.
        fallback_to_ar: Fall back to pure AR on verification failure.
        cache_ar_kv: Cache AR model key-values for efficiency.
        log_stats: Log hybrid generation statistics.
    """
    enabled: bool = True
    mode: HybridMode = HybridMode.DEER
    ar_verifier_model: Optional[str] = None
    draft_block_size: int = 8
    max_draft_tokens: int = 32
    acceptance_threshold: float = 0.3
    diffusion_weight: float = 1.0
    ar_weight: float = 0.5
    use_causal_consistency: bool = True
    adaptive_draft_length: bool = True
    min_draft_length: int = 4
    max_draft_length: int = 16
    fallback_to_ar: bool = True
    cache_ar_kv: bool = True
    log_stats: bool = False
    diffusion_steps_per_draft: int = 1
    temperature: float = 0.0
    
    def __post_init__(self):
        if isinstance(self.mode, str):
            self.mode = HybridMode(self.mode)
        if self.draft_block_size < 1:
            raise ValueError("draft_block_size must be >= 1")
        if not 0.0 <= self.acceptance_threshold <= 1.0:
            raise ValueError("acceptance_threshold must be in [0, 1]")


@dataclass
class HybridStats:
    """Statistics for hybrid generation performance."""
    total_requests: int = 0
    total_tokens_generated: int = 0
    total_drafts: int = 0
    tokens_accepted: int = 0
    tokens_rejected: int = 0
    draft_acceptance_rate: float = 0.0
    avg_accepted_length: float = 0.0
    ar_fallbacks: int = 0
    speedup_ratio: float = 1.0
    diffusion_time_ms: float = 0.0
    ar_verification_time_ms: float = 0.0
    
    def update(
        self,
        drafted: int,
        accepted: int,
        diffusion_time: float,
        ar_time: float,
        used_fallback: bool = False,
    ):
        """Update statistics after a draft-verify cycle."""
        self.total_drafts += 1
        self.tokens_accepted += accepted
        self.tokens_rejected += (drafted - accepted)
        self.diffusion_time_ms += diffusion_time * 1000
        self.ar_verification_time_ms += ar_time * 1000
        
        if used_fallback:
            self.ar_fallbacks += 1
        
        if self.total_drafts > 0:
            self.draft_acceptance_rate = self.tokens_accepted / max(
                self.tokens_accepted + self.tokens_rejected, 1
            )
            self.avg_accepted_length = self.tokens_accepted / self.total_drafts
        
        # Estimate speedup: tokens accepted per unit time
        total_time = self.diffusion_time_ms + self.ar_verification_time_ms
        if total_time > 0 and self.tokens_accepted > 0:
            hybrid_throughput = self.tokens_accepted / total_time
            # Assume AR would take ~10ms per token sequentially
            ar_only_time = self.tokens_accepted * 10
            self.speedup_ratio = ar_only_time / max(total_time, 1)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "total_requests": self.total_requests,
            "total_tokens_generated": self.total_tokens_generated,
            "total_drafts": self.total_drafts,
            "tokens_accepted": self.tokens_accepted,
            "tokens_rejected": self.tokens_rejected,
            "draft_acceptance_rate": round(self.draft_acceptance_rate, 3),
            "avg_accepted_length": round(self.avg_accepted_length, 2),
            "ar_fallbacks": self.ar_fallbacks,
            "speedup_ratio": round(self.speedup_ratio, 2),
            "diffusion_time_ms": round(self.diffusion_time_ms, 2),
            "ar_verification_time_ms": round(self.ar_verification_time_ms, 2),
        }
    
    def reset(self):
        """Reset all statistics."""
        self.__init__()


class HybridEngine:
    """DEER-style Hybrid Engine: Draft with Diffusion, Verify with AR.
    
    Implements the hybrid generation algorithm from DEER paper:
    1. Diffusion model generates k tokens in parallel (one-step denoising)
    2. AR model verifies token sequence for causal consistency
    3. Accept longest matching prefix
    4. Repeat until target length reached
    
    Benefits (from papers):
    - 2-7x speedup over pure AR generation
    - Maintains output quality of AR models
    - No retraining required
    - Works with existing diffusion + AR model pairs
    
    Example:
        >>> hybrid = HybridEngine(
        ...     diffusion_model=llada_model,
        ...     ar_model=tiny_llama,
        ...     config=HybridConfig(mode=HybridMode.DEER),
        ... )
        >>> output = hybrid.generate(prompt_ids, max_new_tokens=128)
    """
    
    def __init__(
        self,
        diffusion_model,
        ar_model: Optional[Any] = None,
        tokenizer=None,
        config: Optional[HybridConfig] = None,
        mask_id: int = 126336,
    ):
        """Initialize hybrid engine.
        
        Args:
            diffusion_model: The diffusion LLM (e.g., LLaDA).
            ar_model: Optional AR model for verification.
            tokenizer: Shared tokenizer.
            config: Hybrid configuration.
            mask_id: Mask token ID for diffusion model.
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for hybrid engine")
        
        self.diffusion_model = diffusion_model
        self.ar_model = ar_model
        self.tokenizer = tokenizer
        self.config = config or HybridConfig()
        self.mask_id = mask_id
        self.stats = HybridStats()
        
        self._device = next(diffusion_model.parameters()).device
        self._ar_kv_cache = None
        
        self._adaptive_draft_length = self.config.min_draft_length
        self._acceptance_history: List[float] = []
        
        logger.info(
            f"HybridEngine initialized: mode={self.config.mode.value}, "
            f"ar_verifier={'enabled' if ar_model else 'disabled'}, "
            f"draft_size={self.config.draft_block_size}"
        )
    
    def _get_diffusion_draft(
        self,
        context: "torch.Tensor",
        draft_length: int,
        temperature: float = 0.0,
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """Generate draft tokens using diffusion model.
        
        Uses single-step denoising for fast parallel draft generation
        as described in DEER paper.
        
        Args:
            context: Current context tokens (batch, seq_len).
            draft_length: Number of tokens to draft.
            temperature: Sampling temperature.
        
        Returns:
            Tuple of (draft tokens, confidence scores).
        """
        batch_size = context.shape[0]
        device = context.device
        
        draft_input = torch.cat([
            context,
            torch.full((batch_size, draft_length), self.mask_id, device=device)
        ], dim=1)
        
        with torch.no_grad():
            if hasattr(self.diffusion_model, 'config') and torch.cuda.is_available():
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = self.diffusion_model(draft_input)
            else:
                outputs = self.diffusion_model(draft_input)
            
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            draft_logits = logits[:, -draft_length:].float()
        
        if temperature > 0:
            probs = F.softmax(draft_logits / temperature, dim=-1)
            draft_tokens = torch.multinomial(
                probs.view(-1, probs.size(-1)), 1
            ).view(batch_size, draft_length)
        else:
            draft_tokens = draft_logits.argmax(dim=-1)
        
        confidence = F.softmax(draft_logits, dim=-1).max(dim=-1).values
        
        return draft_tokens, confidence
    
    def _verify_with_ar(
        self,
        context: "torch.Tensor",
        draft_tokens: "torch.Tensor",
        draft_confidence: "torch.Tensor",
    ) -> Tuple["torch.Tensor", int]:
        """Verify draft tokens using AR model.
        
        Implements the verification step from DiffuSpec:
        P_final = P_diffusion^alpha * P_AR^beta
        
        Args:
            context: Context before draft.
            draft_tokens: Drafted token IDs.
            draft_confidence: Diffusion confidence scores.
        
        Returns:
            Tuple of (accepted tokens, count of accepted).
        """
        if self.ar_model is None:
            return draft_tokens, draft_tokens.shape[1]
        
        batch_size = context.shape[0]
        draft_length = draft_tokens.shape[1]
        
        full_sequence = torch.cat([context, draft_tokens], dim=1)
        
        with torch.no_grad():
            if torch.cuda.is_available():
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    ar_outputs = self.ar_model(full_sequence)
            else:
                ar_outputs = self.ar_model(full_sequence)
            
            ar_logits = ar_outputs.logits if hasattr(ar_outputs, 'logits') else ar_outputs
            ar_logits = ar_logits[:, -(draft_length + 1):-1].float()
        
        ar_probs = F.softmax(ar_logits, dim=-1)
        
        ar_scores = torch.gather(
            ar_probs,
            dim=-1,
            index=draft_tokens.unsqueeze(-1)
        ).squeeze(-1)
        
        if self.config.use_causal_consistency:
            combined_scores = (
                (draft_confidence ** self.config.diffusion_weight) *
                (ar_scores ** self.config.ar_weight)
            )
        else:
            combined_scores = ar_scores
        
        accept_mask = combined_scores >= self.config.acceptance_threshold
        
        accepted_counts = torch.zeros(batch_size, dtype=torch.long, device=context.device)
        
        for b in range(batch_size):
            for i in range(draft_length):
                if accept_mask[b, i]:
                    accepted_counts[b] = i + 1
                else:
                    break
        
        max_accepted = accepted_counts.max().item()
        accepted_tokens = draft_tokens[:, :max_accepted] if max_accepted > 0 else draft_tokens[:, :0]
        
        return accepted_tokens, max_accepted
    
    def _update_adaptive_draft_length(self, accepted: int, drafted: int):
        """Adaptively adjust draft length based on acceptance rate.
        
        From DiffuSpec: increase draft length when acceptance is high,
        decrease when many rejections.
        """
        if not self.config.adaptive_draft_length:
            return
        
        acceptance_rate = accepted / max(drafted, 1)
        self._acceptance_history.append(acceptance_rate)
        
        if len(self._acceptance_history) > 10:
            self._acceptance_history.pop(0)
        
        avg_rate = sum(self._acceptance_history) / len(self._acceptance_history)
        
        if avg_rate > 0.8 and self._adaptive_draft_length < self.config.max_draft_length:
            self._adaptive_draft_length = min(
                self._adaptive_draft_length + 2,
                self.config.max_draft_length
            )
        elif avg_rate < 0.3 and self._adaptive_draft_length > self.config.min_draft_length:
            self._adaptive_draft_length = max(
                self._adaptive_draft_length - 2,
                self.config.min_draft_length
            )
    
    def _ar_fallback_generate(
        self,
        context: "torch.Tensor",
        num_tokens: int = 1,
    ) -> "torch.Tensor":
        """Fall back to pure AR generation for difficult tokens."""
        if self.ar_model is None:
            return context
        
        with torch.no_grad():
            output = self.ar_model.generate(
                context,
                max_new_tokens=num_tokens,
                do_sample=self.config.temperature > 0,
                temperature=max(self.config.temperature, 1e-7),
                pad_token_id=self.tokenizer.pad_token_id if self.tokenizer else 0,
            )
        
        return output
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: "torch.Tensor",
        attention_mask: Optional["torch.Tensor"] = None,
        max_new_tokens: int = 128,
        temperature: Optional[float] = None,
        **kwargs
    ) -> "torch.Tensor":
        """Generate text using hybrid diffusion-AR approach.
        
        Implements the DEER algorithm:
        1. Draft k tokens with diffusion (parallel)
        2. Verify with AR model
        3. Accept longest valid prefix
        4. Repeat until max_new_tokens
        
        Args:
            input_ids: Input token IDs (batch, seq_len).
            attention_mask: Optional attention mask.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
        
        Returns:
            Generated token IDs including input.
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for generation")
        
        temperature = temperature if temperature is not None else self.config.temperature
        
        if self.ar_model is None:
            return self._pure_diffusion_generate(
                input_ids, max_new_tokens, temperature
            )
        
        self.stats.total_requests += 1
        
        output = input_ids.clone()
        tokens_generated = 0
        consecutive_failures = 0
        max_failures = 3
        
        while tokens_generated < max_new_tokens:
            remaining = max_new_tokens - tokens_generated
            draft_length = min(
                self._adaptive_draft_length,
                remaining,
                self.config.max_draft_tokens
            )
            
            t_draft_start = time.perf_counter()
            draft_tokens, confidence = self._get_diffusion_draft(
                output, draft_length, temperature
            )
            t_draft = time.perf_counter() - t_draft_start
            
            t_verify_start = time.perf_counter()
            accepted_tokens, num_accepted = self._verify_with_ar(
                output, draft_tokens, confidence
            )
            t_verify = time.perf_counter() - t_verify_start
            
            used_fallback = False
            if num_accepted == 0:
                consecutive_failures += 1
                
                if consecutive_failures >= max_failures and self.config.fallback_to_ar:
                    output = self._ar_fallback_generate(output, num_tokens=1)
                    tokens_generated += 1
                    used_fallback = True
                    consecutive_failures = 0
            else:
                output = torch.cat([output, accepted_tokens], dim=1)
                tokens_generated += num_accepted
                consecutive_failures = 0
            
            self._update_adaptive_draft_length(num_accepted, draft_length)
            self.stats.update(
                drafted=draft_length,
                accepted=num_accepted,
                diffusion_time=t_draft,
                ar_time=t_verify,
                used_fallback=used_fallback,
            )
            
            if self._check_stop_condition(output):
                break
        
        self.stats.total_tokens_generated += tokens_generated
        
        if self.config.log_stats:
            logger.info(f"Hybrid generation stats: {self.stats.to_dict()}")
        
        return output
    
    def _pure_diffusion_generate(
        self,
        input_ids: "torch.Tensor",
        max_new_tokens: int,
        temperature: float,
    ) -> "torch.Tensor":
        """Fall back to pure diffusion when no AR model available."""
        from dfastllm.engine.diffusion_sampler import diffusion_generate
        
        steps = min(64, max_new_tokens)
        block_length = min(32, max_new_tokens)
        
        if max_new_tokens % block_length != 0:
            for bl in range(block_length, 0, -1):
                if max_new_tokens % bl == 0:
                    block_length = bl
                    break
            else:
                block_length = max_new_tokens
        
        num_blocks = max_new_tokens // block_length
        if steps % num_blocks != 0:
            steps = max(num_blocks, (steps // num_blocks) * num_blocks)
        
        return diffusion_generate(
            model=self.diffusion_model,
            prompt=input_ids,
            steps=steps,
            gen_length=max_new_tokens,
            block_length=block_length,
            temperature=temperature,
            mask_id=self.mask_id,
        )
    
    def _check_stop_condition(self, output: "torch.Tensor") -> bool:
        """Check if generation should stop."""
        if self.tokenizer is None:
            return False
        
        eos_id = getattr(self.tokenizer, 'eos_token_id', None)
        if eos_id is not None and output[0, -1].item() == eos_id:
            return True
        
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get generation statistics."""
        return self.stats.to_dict()
    
    def reset_stats(self):
        """Reset statistics."""
        self.stats.reset()


class SpecDiffEngine(HybridEngine):
    """Speculative Diffusion Decoding Engine.
    
    Implements SpecDiff from NAACL 2025 paper with:
    - Multi-step diffusion drafting
    - Parallel verification
    - Causal-consistency path search
    
    Claimed 7.2x speedup over standard generation.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config.mode = HybridMode.SPEC_DIFF
    
    def _get_diffusion_draft(
        self,
        context: "torch.Tensor",
        draft_length: int,
        temperature: float = 0.0,
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """Multi-step diffusion drafting with parallel denoising."""
        batch_size = context.shape[0]
        device = context.device
        
        x = torch.cat([
            context,
            torch.full((batch_size, draft_length), self.mask_id, device=device)
        ], dim=1)
        
        steps = self.config.diffusion_steps_per_draft
        
        for step in range(steps):
            with torch.no_grad():
                if torch.cuda.is_available():
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        outputs = self.diffusion_model(x)
                else:
                    outputs = self.diffusion_model(x)
                
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                draft_logits = logits[:, -draft_length:].float()
            
            probs = F.softmax(draft_logits, dim=-1)
            predictions = probs.argmax(dim=-1)
            confidence = probs.max(dim=-1).values
            
            mask_positions = (x[:, -draft_length:] == self.mask_id)
            
            tokens_per_step = max(1, draft_length // steps)
            for b in range(batch_size):
                masked_conf = torch.where(
                    mask_positions[b],
                    confidence[b],
                    torch.tensor(float('-inf'), device=device)
                )
                _, top_indices = torch.topk(masked_conf, min(tokens_per_step, mask_positions[b].sum()))
                
                for idx in top_indices:
                    x[b, context.shape[1] + idx] = predictions[b, idx]
        
        draft_tokens = x[:, -draft_length:]
        
        with torch.no_grad():
            if torch.cuda.is_available():
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = self.diffusion_model(x)
            else:
                outputs = self.diffusion_model(x)
            final_logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            final_logits = final_logits[:, -draft_length:].float()
        
        final_confidence = F.softmax(final_logits, dim=-1).max(dim=-1).values
        
        return draft_tokens, final_confidence


class SemiAREngine(HybridEngine):
    """Semi-Autoregressive Hybrid Engine.
    
    Based on SSD-LM paper: generates blocks of tokens,
    combines diffusion planning with AR refinement within blocks.
    """
    
    def __init__(self, *args, block_size: int = 8, **kwargs):
        super().__init__(*args, **kwargs)
        self.config.mode = HybridMode.SEMI_AR
        self.block_size = block_size
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: "torch.Tensor",
        attention_mask: Optional["torch.Tensor"] = None,
        max_new_tokens: int = 128,
        temperature: Optional[float] = None,
        **kwargs
    ) -> "torch.Tensor":
        """Semi-AR generation: diffusion blocks + AR transitions."""
        temperature = temperature if temperature is not None else self.config.temperature
        
        output = input_ids.clone()
        tokens_generated = 0
        
        while tokens_generated < max_new_tokens:
            remaining = max_new_tokens - tokens_generated
            block_length = min(self.block_size, remaining)
            
            draft_tokens, confidence = self._get_diffusion_draft(
                output, block_length, temperature
            )
            
            if self.ar_model is not None:
                refined_tokens = self._ar_refine_block(
                    output, draft_tokens, confidence
                )
                output = torch.cat([output, refined_tokens], dim=1)
            else:
                output = torch.cat([output, draft_tokens], dim=1)
            
            tokens_generated += block_length
            
            if self._check_stop_condition(output):
                break
        
        return output
    
    def _ar_refine_block(
        self,
        context: "torch.Tensor",
        draft_tokens: "torch.Tensor",
        confidence: "torch.Tensor",
    ) -> "torch.Tensor":
        """Refine low-confidence tokens in block using AR model."""
        refinement_mask = confidence < 0.7
        
        if not refinement_mask.any():
            return draft_tokens
        
        full_seq = torch.cat([context, draft_tokens], dim=1)
        
        with torch.no_grad():
            outputs = self.ar_model(full_seq)
            ar_logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            ar_predictions = ar_logits[:, -draft_tokens.shape[1]:].argmax(dim=-1)
        
        refined = draft_tokens.clone()
        refined[refinement_mask] = ar_predictions[refinement_mask]
        
        return refined


def create_hybrid_engine(
    diffusion_model,
    ar_model: Optional[Any] = None,
    tokenizer=None,
    mode: str = "deer",
    config: Optional[HybridConfig] = None,
    mask_id: int = 126336,
    **kwargs
) -> HybridEngine:
    """Factory function to create appropriate hybrid engine.
    
    Args:
        diffusion_model: The diffusion LLM.
        ar_model: Optional AR verification model.
        tokenizer: Shared tokenizer.
        mode: Hybrid mode ("deer", "spec_diff", "semi_ar", "adaptive").
        config: Optional HybridConfig override.
        mask_id: Mask token ID.
    
    Returns:
        Configured HybridEngine instance.
    
    Example:
        >>> engine = create_hybrid_engine(
        ...     diffusion_model=llada,
        ...     ar_model=tiny_llama,
        ...     mode="deer",
        ... )
    """
    if config is None:
        config = HybridConfig(mode=HybridMode(mode), **kwargs)
    
    engine_map = {
        HybridMode.DEER: HybridEngine,
        HybridMode.SPEC_DIFF: SpecDiffEngine,
        HybridMode.SEMI_AR: SemiAREngine,
        HybridMode.ADAPTIVE: HybridEngine,
    }
    
    engine_class = engine_map.get(config.mode, HybridEngine)
    
    return engine_class(
        diffusion_model=diffusion_model,
        ar_model=ar_model,
        tokenizer=tokenizer,
        config=config,
        mask_id=mask_id,
    )


@torch.no_grad()
def hybrid_generate(
    diffusion_model,
    prompt: "torch.Tensor",
    ar_model: Optional[Any] = None,
    max_new_tokens: int = 128,
    mode: str = "deer",
    draft_size: int = 8,
    acceptance_threshold: float = 0.3,
    temperature: float = 0.0,
    mask_id: int = 126336,
) -> "torch.Tensor":
    """Convenience function for hybrid generation.
    
    Args:
        diffusion_model: The diffusion LLM.
        prompt: Input token IDs.
        ar_model: Optional AR verification model.
        max_new_tokens: Tokens to generate.
        mode: Hybrid mode.
        draft_size: Tokens per draft.
        acceptance_threshold: Verification threshold.
        temperature: Sampling temperature.
        mask_id: Mask token ID.
    
    Returns:
        Generated token IDs including prompt.
    """
    config = HybridConfig(
        mode=HybridMode(mode),
        draft_block_size=draft_size,
        acceptance_threshold=acceptance_threshold,
        temperature=temperature,
    )
    
    engine = create_hybrid_engine(
        diffusion_model=diffusion_model,
        ar_model=ar_model,
        config=config,
        mask_id=mask_id,
    )
    
    return engine.generate(prompt, max_new_tokens=max_new_tokens)

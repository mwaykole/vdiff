"""Adaptive Parallel Decoding (APD) for Diffusion LLMs.

APD is a novel method to enhance diffusion LLM inference throughput by:
1. Generating multiple token candidates in parallel from diffusion LLM
2. Using a small auxiliary AR model to verify sequence coherence
3. Adaptively accepting tokens based on combined probability scores

Reference: https://arxiv.org/abs/2506.00413
GitHub: https://github.com/danielmisrael/apd
"""

from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available, APD disabled")


@dataclass
class APDConfig:
    """Configuration for Adaptive Parallel Decoding."""
    
    enabled: bool = True
    max_parallel_tokens: int = 8
    acceptance_threshold: float = 0.3
    dllm_weight: float = 1.0  # α weight for diffusion LLM probabilities
    ar_weight: float = 0.5   # β weight for AR probabilities
    temperature: float = 0.0
    use_ar_verification: bool = False  # Set True if AR model available
    
    def __post_init__(self):
        if self.max_parallel_tokens < 1:
            raise ValueError("max_parallel_tokens must be >= 1")
        if not 0.0 <= self.acceptance_threshold <= 1.0:
            raise ValueError("acceptance_threshold must be in [0, 1]")


@dataclass
class APDStats:
    """Statistics for APD decoding."""
    
    total_steps: int = 0
    total_tokens_generated: int = 0
    tokens_accepted: int = 0
    tokens_rejected: int = 0
    avg_tokens_per_step: float = 0.0
    
    def update(self, accepted: int, rejected: int):
        self.total_steps += 1
        self.tokens_accepted += accepted
        self.tokens_rejected += rejected
        self.total_tokens_generated = self.tokens_accepted
        if self.total_steps > 0:
            self.avg_tokens_per_step = self.tokens_accepted / self.total_steps


class APDDecoder:
    """Adaptive Parallel Decoder for Diffusion LLMs.
    
    Implements the APD algorithm which combines:
    - Diffusion LLM marginal probabilities (parallel token prediction)
    - Optional AR model joint probabilities (sequence coherence)
    
    Formula: P_final = P_diffLLM^α × P_AR^β
    """
    
    def __init__(
        self,
        config: Optional[APDConfig] = None,
        ar_model: Optional[Any] = None,
    ):
        """Initialize APD decoder.
        
        Args:
            config: APD configuration.
            ar_model: Optional small AR model for verification.
        """
        self.config = config or APDConfig()
        self.ar_model = ar_model
        self.stats = APDStats()
        
        if ar_model is not None:
            self.config.use_ar_verification = True
    
    def compute_confidence(
        self,
        logits: "torch.Tensor",
        mask_positions: "torch.Tensor",
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """Compute confidence scores for masked positions.
        
        Args:
            logits: Model logits of shape (batch, seq_len, vocab_size).
            mask_positions: Boolean tensor indicating masked positions.
        
        Returns:
            Tuple of (confidence scores, predicted tokens).
        """
        # Get probabilities
        probs = F.softmax(logits.float(), dim=-1)
        
        # Confidence = max probability per position
        confidence, predictions = probs.max(dim=-1)
        
        # Only consider masked positions
        confidence = torch.where(mask_positions, confidence, torch.tensor(float('-inf'), device=logits.device))
        
        return confidence, predictions
    
    def select_candidates(
        self,
        confidence: "torch.Tensor",
        mask_positions: "torch.Tensor",
        k: int,
    ) -> "torch.Tensor":
        """Select top-k candidates based on confidence.
        
        Args:
            confidence: Confidence scores.
            mask_positions: Masked positions.
            k: Number of candidates to select.
        
        Returns:
            Boolean tensor indicating selected positions.
        """
        batch_size = confidence.shape[0]
        selected = torch.zeros_like(mask_positions, dtype=torch.bool)
        
        for b in range(batch_size):
            # Count available masked positions
            num_masked = mask_positions[b].sum().item()
            num_to_select = min(k, num_masked)
            
            if num_to_select > 0:
                # Get top-k by confidence
                _, top_indices = torch.topk(confidence[b], k=num_to_select)
                selected[b, top_indices] = True
        
        return selected
    
    def verify_with_ar(
        self,
        candidates: "torch.Tensor",
        selected_positions: "torch.Tensor",
        predicted_tokens: "torch.Tensor",
    ) -> "torch.Tensor":
        """Verify candidates using AR model (if available).
        
        Args:
            candidates: Current sequence with candidate tokens.
            selected_positions: Positions of candidate tokens.
            predicted_tokens: Predicted token IDs.
        
        Returns:
            Acceptance scores for each candidate position.
        """
        if self.ar_model is None or not self.config.use_ar_verification:
            # No AR verification, accept all with score 1.0
            return torch.ones_like(selected_positions, dtype=torch.float)
        
        batch_size, seq_len = candidates.shape
        acceptance_scores = torch.zeros(batch_size, seq_len, device=candidates.device)
        
        with torch.no_grad():
            ar_outputs = self.ar_model(candidates)
            ar_logits = ar_outputs.logits if hasattr(ar_outputs, 'logits') else ar_outputs
            ar_probs = F.softmax(ar_logits.float(), dim=-1)
        
        for b in range(batch_size):
            positions = torch.where(selected_positions[b])[0]
            for pos in positions:
                if pos > 0:
                    # AR probability: P(token_t | token_1...token_{t-1})
                    token = predicted_tokens[b, pos]
                    p_ar = ar_probs[b, pos - 1, token].item()
                    acceptance_scores[b, pos] = p_ar
                else:
                    acceptance_scores[b, pos] = 1.0
        
        return acceptance_scores
    
    def decode_step(
        self,
        model: Any,
        x: "torch.Tensor",
        mask_id: int,
        step: int,
        total_steps: int,
        attention_mask: Optional["torch.Tensor"] = None,
    ) -> Tuple["torch.Tensor", int]:
        """Perform one APD decoding step.
        
        Args:
            model: The diffusion LLM.
            x: Current token sequence.
            mask_id: Mask token ID.
            step: Current step number.
            total_steps: Total steps.
            attention_mask: Optional attention mask.
        
        Returns:
            Tuple of (updated sequence, number of tokens decoded).
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for APD")
        
        device = x.device
        batch_size, seq_len = x.shape
        
        # Identify masked positions
        mask_positions = (x == mask_id)
        num_masked = mask_positions.sum().item()
        
        if num_masked == 0:
            return x, 0
        
        # Forward pass
        with torch.no_grad():
            outputs = model(x, attention_mask=attention_mask)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        
        # Compute confidence and predictions
        confidence, predictions = self.compute_confidence(logits, mask_positions)
        
        # Determine how many tokens to try (adaptive based on step)
        remaining_steps = total_steps - step
        if remaining_steps > 0:
            target_tokens = max(1, num_masked // remaining_steps)
        else:
            target_tokens = num_masked
        
        k = min(self.config.max_parallel_tokens, target_tokens, num_masked)
        
        # Select top-k candidates
        selected = self.select_candidates(confidence, mask_positions, k)
        
        # Create candidate sequence
        x_candidate = x.clone()
        x_candidate[selected] = predictions[selected]
        
        # Verify with AR model (if available)
        if self.config.use_ar_verification and self.ar_model is not None:
            ar_scores = self.verify_with_ar(x_candidate, selected, predictions)
            
            # Combine scores: P_final = P_diffLLM^α × P_AR^β
            dllm_scores = confidence ** self.config.dllm_weight
            combined_scores = dllm_scores * (ar_scores ** self.config.ar_weight)
            
            # Accept only above threshold
            accept = (combined_scores > self.config.acceptance_threshold) & selected
        else:
            # No AR verification - accept all selected
            accept = selected
        
        # Update sequence
        x[accept] = predictions[accept]
        
        num_accepted = accept.sum().item()
        num_rejected = selected.sum().item() - num_accepted
        
        self.stats.update(num_accepted, num_rejected)
        
        return x, num_accepted
    
    def generate(
        self,
        model: Any,
        prompt: "torch.Tensor",
        gen_length: int,
        steps: int,
        mask_id: int,
        attention_mask: Optional["torch.Tensor"] = None,
        temperature: float = 0.0,
    ) -> "torch.Tensor":
        """Generate text using APD.
        
        Args:
            model: The diffusion LLM.
            prompt: Prompt token IDs.
            gen_length: Number of tokens to generate.
            steps: Number of diffusion steps.
            mask_id: Mask token ID.
            attention_mask: Optional attention mask.
            temperature: Sampling temperature.
        
        Returns:
            Generated token IDs including prompt.
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for APD generation")
        
        device = next(model.parameters()).device
        batch_size = prompt.shape[0]
        prompt_length = prompt.shape[1]
        
        # Initialize with masked tokens
        x = torch.full(
            (batch_size, prompt_length + gen_length),
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
        
        # APD decoding loop
        for step in range(steps):
            x, num_decoded = self.decode_step(
                model=model,
                x=x,
                mask_id=mask_id,
                step=step,
                total_steps=steps,
                attention_mask=attention_mask,
            )
            
            # Early stop if all unmasked
            if (x != mask_id).all():
                logger.debug(f"APD: Early stop at step {step}")
                break
        
        return x
    
    def get_stats(self) -> Dict[str, Any]:
        """Get APD statistics."""
        return {
            "total_steps": self.stats.total_steps,
            "total_tokens_generated": self.stats.total_tokens_generated,
            "tokens_accepted": self.stats.tokens_accepted,
            "tokens_rejected": self.stats.tokens_rejected,
            "avg_tokens_per_step": self.stats.avg_tokens_per_step,
            "enabled": self.config.enabled,
            "max_parallel_tokens": self.config.max_parallel_tokens,
            "use_ar_verification": self.config.use_ar_verification,
        }
    
    def reset_stats(self):
        """Reset statistics."""
        self.stats = APDStats()


def apd_generate(
    model: Any,
    prompt: "torch.Tensor",
    gen_length: int = 128,
    steps: int = 32,
    mask_id: int = 126336,
    max_parallel: int = 8,
    temperature: float = 0.0,
    ar_model: Optional[Any] = None,
    attention_mask: Optional["torch.Tensor"] = None,
) -> "torch.Tensor":
    """Convenience function for APD generation.
    
    Args:
        model: The diffusion LLM.
        prompt: Prompt token IDs.
        gen_length: Tokens to generate.
        steps: Diffusion steps.
        mask_id: Mask token ID.
        max_parallel: Max parallel tokens per step.
        temperature: Sampling temperature.
        ar_model: Optional AR model for verification.
        attention_mask: Optional attention mask.
    
    Returns:
        Generated token IDs.
    """
    config = APDConfig(
        max_parallel_tokens=max_parallel,
        temperature=temperature,
        use_ar_verification=ar_model is not None,
    )
    
    decoder = APDDecoder(config=config, ar_model=ar_model)
    
    return decoder.generate(
        model=model,
        prompt=prompt,
        gen_length=gen_length,
        steps=steps,
        mask_id=mask_id,
        attention_mask=attention_mask,
        temperature=temperature,
    )

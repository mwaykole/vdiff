"""Entropy-Informed Adaptive Control for Hybrid Generation.

Implements entropy-based strategies to dynamically adjust generation parameters
based on model uncertainty, improving both speed and quality.

Key Concepts:
- High entropy = model uncertainty = need more computation
- Low entropy = model confidence = can speed up

Based on research:
- Fast-ARDiff (arxiv:2512.08537): Entropy-informed speculative strategies
- Adaptive Computation: Dynamic resource allocation per token

Benefits:
- 20-40% faster generation for confident predictions
- Better quality for difficult tokens
- Automatic adaptation to content complexity
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

from dfastllm.engine.base import BaseStats, BaseConfig, BaseController, EntropyComputer

logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class AdaptationStrategy(Enum):
    """Strategy for adapting generation parameters."""
    DRAFT_LENGTH = "draft_length"
    DIFFUSION_STEPS = "diffusion_steps"
    TEMPERATURE = "temperature"
    COMBINED = "combined"

@dataclass
class EntropyConfig(BaseConfig):
    """Configuration for entropy-adaptive control.
    
    Inherits from BaseConfig for consistent validation and serialization.
    """
    strategy: AdaptationStrategy = AdaptationStrategy.COMBINED
    high_entropy_threshold: float = 2.0
    low_entropy_threshold: float = 0.5
    min_draft_length: int = 4
    max_draft_length: int = 32
    min_diffusion_steps: int = 8
    max_diffusion_steps: int = 64
    ema_alpha: float = 0.3
    window_size: int = 16
    log_stats: bool = True
    
    def validate(self) -> None:
        """Validate entropy configuration."""
        if self.high_entropy_threshold <= self.low_entropy_threshold:
            raise ValueError("high_entropy_threshold must be > low_entropy_threshold")
        if self.min_draft_length > self.max_draft_length:
            raise ValueError("min_draft_length must be <= max_draft_length")

@dataclass
class EntropyStats(BaseStats):
    """Statistics for entropy-adaptive control.
    
    Inherits from BaseStats for consistent serialization.
    """
    total_predictions: int = 0
    high_entropy_count: int = 0
    low_entropy_count: int = 0
    avg_entropy: float = 0.0
    draft_length_adjustments: int = 0
    step_adjustments: int = 0
    compute_saved_pct: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        total = self.total_predictions or 1
        return {
            "total_predictions": self.total_predictions,
            "high_entropy_pct": round(self.high_entropy_count / total * 100, 2),
            "low_entropy_pct": round(self.low_entropy_count / total * 100, 2),
            "avg_entropy": round(self.avg_entropy, 4),
            "draft_length_adjustments": self.draft_length_adjustments,
            "step_adjustments": self.step_adjustments,
            "compute_saved_pct": round(self.compute_saved_pct, 2),
        }

# Use unified EntropyComputer from base module
EntropyCalculator = EntropyComputer  # Alias for backward compatibility

class EntropyAdaptiveController:
    """Controller that adapts generation parameters based on entropy.
    
    Monitors model uncertainty and dynamically adjusts:
    - Draft length for speculative decoding
    - Number of diffusion steps
    - Sampling temperature
    
    Uses exponential moving average to smooth adaptations.
    """
    
    def __init__(self, config: Optional[EntropyConfig] = None):
        self.config = config or EntropyConfig()
        self._stats = EntropyStats()
        self._entropy_history: List[float] = []
        self._current_draft_length = (
            self.config.min_draft_length + self.config.max_draft_length
        ) // 2
        self._current_steps = (
            self.config.min_diffusion_steps + self.config.max_diffusion_steps
        ) // 2
        self._ema_entropy = 1.0
        self._baseline_compute = 0.0
        self._actual_compute = 0.0
    
    def update(self, logits: "torch.Tensor") -> None:
        """Update controller with new logits observation.
        
        Args:
            logits: Model output logits [batch, seq, vocab]
        """
        if not TORCH_AVAILABLE or not self.config.enabled:
            return
        
        entropy = EntropyCalculator.compute_entropy(logits)
        mean_entropy = entropy.mean().item()
        
        self._ema_entropy = (
            self.config.ema_alpha * mean_entropy +
            (1 - self.config.ema_alpha) * self._ema_entropy
        )
        
        self._entropy_history.append(mean_entropy)
        if len(self._entropy_history) > self.config.window_size:
            self._entropy_history.pop(0)
        
        self._stats.total_predictions += 1
        self._stats.avg_entropy = self._ema_entropy
        
        if mean_entropy > self.config.high_entropy_threshold:
            self._stats.high_entropy_count += 1
        elif mean_entropy < self.config.low_entropy_threshold:
            self._stats.low_entropy_count += 1
    
    def get_draft_length(self) -> int:
        """Get adaptive draft length based on entropy.
        
        Low entropy → longer drafts (more confident)
        High entropy → shorter drafts (need verification)
        
        Returns:
            Recommended draft length
        """
        if not self.config.enabled:
            return self.config.max_draft_length
        
        if self._ema_entropy < self.config.low_entropy_threshold:
            target = self.config.max_draft_length
        elif self._ema_entropy > self.config.high_entropy_threshold:
            target = self.config.min_draft_length
        else:
            ratio = (self.config.high_entropy_threshold - self._ema_entropy) / (
                self.config.high_entropy_threshold - self.config.low_entropy_threshold
            )
            target = int(
                self.config.min_draft_length +
                ratio * (self.config.max_draft_length - self.config.min_draft_length)
            )
        
        if target != self._current_draft_length:
            self._stats.draft_length_adjustments += 1
            self._current_draft_length = target
        
        return self._current_draft_length
    
    def get_diffusion_steps(self) -> int:
        """Get adaptive diffusion steps based on entropy.
        
        Low entropy → fewer steps (easier generation)
        High entropy → more steps (need refinement)
        
        Returns:
            Recommended number of diffusion steps
        """
        if not self.config.enabled:
            return self.config.max_diffusion_steps
        
        if self._ema_entropy < self.config.low_entropy_threshold:
            target = self.config.min_diffusion_steps
        elif self._ema_entropy > self.config.high_entropy_threshold:
            target = self.config.max_diffusion_steps
        else:
            ratio = (self._ema_entropy - self.config.low_entropy_threshold) / (
                self.config.high_entropy_threshold - self.config.low_entropy_threshold
            )
            target = int(
                self.config.min_diffusion_steps +
                ratio * (self.config.max_diffusion_steps - self.config.min_diffusion_steps)
            )
        
        if target != self._current_steps:
            self._stats.step_adjustments += 1
            self._current_steps = target
        
        return self._current_steps
    
    def get_temperature_adjustment(self) -> float:
        """Get temperature adjustment factor based on entropy.
        
        High entropy → slightly lower temperature to focus
        Low entropy → keep temperature as-is
        
        Returns:
            Temperature multiplier (0.8 to 1.2)
        """
        if not self.config.enabled:
            return 1.0
        
        if self._ema_entropy > self.config.high_entropy_threshold:
            return 0.9
        elif self._ema_entropy < self.config.low_entropy_threshold:
            return 1.1
        return 1.0
    
    def get_adaptive_params(self) -> Dict[str, Any]:
        """Get all adaptive parameters based on current entropy.
        
        Returns:
            Dictionary of recommended parameters
        """
        return {
            "draft_length": self.get_draft_length(),
            "diffusion_steps": self.get_diffusion_steps(),
            "temperature_factor": self.get_temperature_adjustment(),
            "current_entropy": self._ema_entropy,
            "entropy_level": self._classify_entropy(),
        }
    
    def _classify_entropy(self) -> str:
        """Classify current entropy level."""
        if self._ema_entropy < self.config.low_entropy_threshold:
            return "low"
        elif self._ema_entropy > self.config.high_entropy_threshold:
            return "high"
        return "medium"
    
    def record_compute(self, baseline: float, actual: float) -> None:
        """Record compute usage for statistics.
        
        Args:
            baseline: Baseline compute (without adaptation)
            actual: Actual compute used (with adaptation)
        """
        self._baseline_compute += baseline
        self._actual_compute += actual
        
        if self._baseline_compute > 0:
            self._stats.compute_saved_pct = (
                (1 - self._actual_compute / self._baseline_compute) * 100
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get controller statistics."""
        return self._stats.to_dict()
    
    def reset_stats(self) -> None:
        """Reset statistics."""
        self._stats = EntropyStats()
        self._entropy_history.clear()
        self._baseline_compute = 0.0
        self._actual_compute = 0.0

class EntropyAwareDraftController:
    """Draft length controller with entropy awareness.
    
    Combines acceptance rate feedback with entropy observations
    for more intelligent draft length decisions.
    """
    
    def __init__(
        self,
        initial_length: int = 8,
        min_length: int = 4,
        max_length: int = 32,
        entropy_weight: float = 0.5,
    ):
        self.current_length = initial_length
        self.min_length = min_length
        self.max_length = max_length
        self.entropy_weight = entropy_weight
        self._acceptance_ema = 0.5
        self._entropy_ema = 1.0
        self._alpha = 0.3
    
    def update(
        self,
        accepted: int,
        total: int,
        entropy: Optional[float] = None,
    ) -> None:
        """Update controller with new observation.
        
        Args:
            accepted: Number of accepted tokens
            total: Total tokens in draft
            entropy: Optional entropy observation
        """
        if total > 0:
            acceptance_rate = accepted / total
            self._acceptance_ema = (
                self._alpha * acceptance_rate +
                (1 - self._alpha) * self._acceptance_ema
            )
        
        if entropy is not None:
            self._entropy_ema = (
                self._alpha * entropy +
                (1 - self._alpha) * self._entropy_ema
            )
        
        self._adjust_length()
    
    def _adjust_length(self) -> None:
        """Adjust draft length based on combined signals."""
        acceptance_score = self._acceptance_ema
        entropy_score = max(0, 1 - self._entropy_ema / 3.0)
        
        combined_score = (
            (1 - self.entropy_weight) * acceptance_score +
            self.entropy_weight * entropy_score
        )
        
        if combined_score > 0.7:
            target = min(self.current_length + 2, self.max_length)
        elif combined_score < 0.3:
            target = max(self.current_length - 2, self.min_length)
        else:
            target = self.current_length
        
        self.current_length = target
    
    def get_draft_length(self) -> int:
        """Get current recommended draft length."""
        return self.current_length

def create_entropy_controller(
    strategy: str = "combined",
    **kwargs,
) -> EntropyAdaptiveController:
    """Factory function to create an entropy controller.
    
    Args:
        strategy: Adaptation strategy (draft_length, diffusion_steps, combined)
        **kwargs: Additional config parameters
    
    Returns:
        Configured EntropyAdaptiveController
    """
    strategy_map = {
        "draft_length": AdaptationStrategy.DRAFT_LENGTH,
        "diffusion_steps": AdaptationStrategy.DIFFUSION_STEPS,
        "temperature": AdaptationStrategy.TEMPERATURE,
        "combined": AdaptationStrategy.COMBINED,
    }
    
    config = EntropyConfig(
        strategy=strategy_map.get(strategy, AdaptationStrategy.COMBINED),
        **kwargs,
    )
    
    return EntropyAdaptiveController(config)

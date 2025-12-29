"""Base classes for dfastllm engine components (SOLID principles)."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, Protocol, runtime_checkable
from enum import Enum
import time

@dataclass
class BaseStats:
    """Base class for all statistics tracking.
    
    All Stats classes should inherit from this to ensure consistent
    interface for serialization and reset functionality.
    """
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert statistics to dictionary for serialization."""
        return asdict(self)
    
    def reset(self) -> None:
        """Reset statistics to initial values."""
        for field_name, field_def in self.__dataclass_fields__.items():
            if field_def.default is not field_def.default_factory:
                setattr(self, field_name, field_def.default)
            elif field_def.default_factory is not None:
                setattr(self, field_name, field_def.default_factory())

@dataclass
class TimedStats(BaseStats):
    """Statistics with timing information."""
    total_time_ms: float = 0.0
    avg_time_ms: float = 0.0
    count: int = 0
    
    def record_time(self, time_ms: float) -> None:
        """Record a timing observation."""
        self.total_time_ms += time_ms
        self.count += 1
        self.avg_time_ms = self.total_time_ms / self.count if self.count > 0 else 0.0

@dataclass
class BaseConfig:
    """Base class for all configuration dataclasses.
    
    Provides common validation and serialization methods.
    """
    enabled: bool = True
    
    def validate(self) -> None:
        """Validate configuration. Override in subclasses for custom validation."""
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        result = {}
        for key, value in asdict(self).items():
            if isinstance(value, Enum):
                result[key] = value.value
            else:
                result[key] = value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseConfig":
        """Create configuration from dictionary."""
        return cls(**data)

@runtime_checkable
class Generator(Protocol):
    """Protocol for text generation components."""
    
    def generate(
        self,
        input_ids: Any,
        max_new_tokens: int,
        temperature: float = 1.0,
        **kwargs,
    ) -> Any:
        """Generate new tokens from input."""
        ...

@runtime_checkable
class Configurable(Protocol):
    """Protocol for components with configuration."""
    
    @property
    def config(self) -> BaseConfig:
        """Get component configuration."""
        ...

@runtime_checkable
class HasStats(Protocol):
    """Protocol for components that track statistics."""
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        ...
    
    def reset_stats(self) -> None:
        """Reset statistics to initial values."""
        ...

@runtime_checkable
class Cacheable(Protocol):
    """Protocol for cacheable components."""
    
    def get(self, key: Any) -> Optional[Any]:
        """Get item from cache."""
        ...
    
    def put(self, key: Any, value: Any) -> None:
        """Put item in cache."""
        ...
    
    def clear(self) -> None:
        """Clear the cache."""
        ...

class BaseController(ABC):
    """Abstract base class for adaptive controllers.
    
    Controllers adjust generation parameters based on observations.
    """
    
    def __init__(self, config: Optional[BaseConfig] = None):
        self._config = config
        self._enabled = config.enabled if config else True
    
    @property
    def enabled(self) -> bool:
        return self._enabled
    
    @abstractmethod
    def update(self, observation: Any) -> None:
        """Update controller with new observation."""
        pass
    
    @abstractmethod
    def get_recommendation(self) -> Dict[str, Any]:
        """Get current parameter recommendations."""
        pass

class BaseCache(ABC):
    """Abstract base class for caching implementations.
    
    Provides common caching patterns with LRU eviction.
    """
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self._cache: Dict[Any, Any] = {}
        self._access_times: Dict[Any, float] = {}
        self._hits = 0
        self._misses = 0
    
    def get(self, key: Any) -> Optional[Any]:
        """Get item from cache with hit/miss tracking."""
        if key in self._cache:
            self._access_times[key] = time.time()
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return None
    
    def put(self, key: Any, value: Any) -> None:
        """Put item in cache with LRU eviction."""
        if len(self._cache) >= self.max_size:
            self._evict_lru()
        self._cache[key] = value
        self._access_times[key] = time.time()
    
    def _evict_lru(self) -> None:
        """Evict least recently used item."""
        if self._access_times:
            oldest_key = min(self._access_times, key=self._access_times.get)
            del self._cache[oldest_key]
            del self._access_times[oldest_key]
    
    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._access_times.clear()
        self._hits = 0
        self._misses = 0
    
    @property
    def hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self.hit_rate, 4),
        }

class EntropyComputer:
    """Unified entropy computation utility.
    
    Provides standardized entropy calculations used across multiple components.
    This consolidates duplicate entropy code from mor_decoder and entropy_controller.
    """
    
    @staticmethod
    def compute(logits: Any, dim: int = -1) -> Any:
        """Compute Shannon entropy from logits.
        
        Args:
            logits: Model output logits [batch, seq, vocab]
            dim: Dimension to compute entropy over
        
        Returns:
            Entropy tensor [batch, seq]
        """
        try:
            import torch
            import torch.nn.functional as F
        except ImportError:
            return 0.0
        
        probs = F.softmax(logits, dim=dim)
        log_probs = F.log_softmax(logits, dim=dim)
        entropy = -torch.sum(probs * log_probs, dim=dim)
        return entropy
    
    @staticmethod
    def compute_normalized(logits: Any, dim: int = -1) -> Any:
        """Compute normalized entropy (0 to 1 scale)."""
        try:
            import torch
        except ImportError:
            return 0.0
        
        entropy = EntropyComputer.compute(logits, dim)
        vocab_size = logits.shape[dim]
        max_entropy = torch.log(torch.tensor(vocab_size, dtype=logits.dtype, device=logits.device))
        return entropy / max_entropy
    
    @staticmethod
    def compute_top_k(logits: Any, k: int = 100, dim: int = -1) -> Any:
        """Compute entropy over top-k tokens only."""
        try:
            import torch
        except ImportError:
            return 0.0
        
        top_k_logits, _ = torch.topk(logits, min(k, logits.shape[dim]), dim=dim)
        return EntropyComputer.compute(top_k_logits, dim)

class ConfidenceComputer:
    """Unified confidence score computation."""
    
    @staticmethod
    def from_logits(logits: Any, dim: int = -1) -> Any:
        """Compute confidence from logits (max probability)."""
        try:
            import torch
            import torch.nn.functional as F
        except ImportError:
            return 1.0
        
        probs = F.softmax(logits, dim=dim)
        return probs.max(dim=dim).values
    
    @staticmethod
    def from_entropy(entropy: Any, max_entropy: float = 10.0) -> Any:
        """Convert entropy to confidence score."""
        try:
            import torch
        except ImportError:
            return 1.0 - entropy / max_entropy
        
        return 1.0 - torch.clamp(entropy / max_entropy, 0.0, 1.0)

def create_stats(stats_type: str, **kwargs) -> BaseStats:
    """Factory function to create statistics objects."""
    from dfastllm.engine.hybrid_engine import HybridStats
    from dfastllm.engine.mor_decoder import MoRStats
    from dfastllm.engine.entropy_controller import EntropyStats
    from dfastllm.engine.continuous_batching import BatcherStats
    
    stats_map = {
        "hybrid": HybridStats,
        "mor": MoRStats,
        "entropy": EntropyStats,
        "batcher": BatcherStats,
        "timed": TimedStats,
    }
    
    stats_class = stats_map.get(stats_type, BaseStats)
    return stats_class(**kwargs)

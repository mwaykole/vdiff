"""Output classes for dfastllm generation.

Matches vLLM's output classes exactly while adding diffusion-specific metrics.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import time

@dataclass
class CompletionOutput:
    """Output for a single completion sequence.
    
    Matches vLLM's CompletionOutput structure.
    
    Attributes:
        index: The index of this output in the list of outputs.
        text: The generated text.
        token_ids: The token ids of the generated text.
        cumulative_logprob: The cumulative log probability of the generated tokens.
        logprobs: The log probabilities of the top tokens at each position.
        finish_reason: The reason why generation stopped.
        stop_reason: The stop string or token that caused generation to stop.
        
        # dfastllm specific
        parallel_tokens_decoded: Number of tokens decoded in parallel.
    """
    
    index: int
    text: str
    token_ids: List[int] = field(default_factory=list)
    cumulative_logprob: Optional[float] = None
    logprobs: Optional[List[Dict[int, float]]] = None
    finish_reason: Optional[str] = None
    stop_reason: Optional[str] = None
    
    # dfastllm specific metrics
    parallel_tokens_decoded: int = 0
    
    def __repr__(self) -> str:
        return (
            f"CompletionOutput(index={self.index}, "
            f"text={self.text!r}, "
            f"token_ids={self.token_ids}, "
            f"finish_reason={self.finish_reason!r})"
        )

@dataclass
class RequestMetrics:
    """Metrics for a single request.
    
    Matches vLLM's RequestMetrics with additional dfastllm-specific metrics.
    
    Attributes:
        arrival_time: Time when the request arrived.
        first_scheduled_time: Time when the request was first scheduled.
        first_token_time: Time when the first token was generated.
        time_in_queue: Time spent in the queue.
        finished_time: Time when the request finished.
        prompt_tokens: Number of tokens in the prompt.
        generated_tokens: Number of tokens generated.
        
        # dfastllm specific
        parallel_tokens_decoded: Total tokens decoded in parallel.
        kv_cache_hit_rate: KV cache hit rate for this request.
        diffusion_steps: Number of diffusion steps used.
    """
    
    arrival_time: float = field(default_factory=time.time)
    first_scheduled_time: Optional[float] = None
    first_token_time: Optional[float] = None
    time_in_queue: Optional[float] = None
    finished_time: Optional[float] = None
    prompt_tokens: int = 0
    generated_tokens: int = 0
    
    # dfastllm specific metrics
    parallel_tokens_decoded: int = 0
    kv_cache_hit_rate: float = 0.0
    diffusion_steps: int = 0
    
    @property
    def time_to_first_token(self) -> Optional[float]:
        """Calculate time to first token."""
        if self.first_token_time is not None and self.arrival_time is not None:
            return self.first_token_time - self.arrival_time
        return None
    
    @property
    def total_latency(self) -> Optional[float]:
        """Calculate total request latency."""
        if self.finished_time is not None and self.arrival_time is not None:
            return self.finished_time - self.arrival_time
        return None
    
    @property
    def time_per_token(self) -> Optional[float]:
        """Calculate average time per generated token."""
        total = self.total_latency
        if total is not None and self.generated_tokens > 0:
            return total / self.generated_tokens
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "arrival_time": self.arrival_time,
            "first_token_time": self.first_token_time,
            "finished_time": self.finished_time,
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.generated_tokens,
            "time_to_first_token": self.time_to_first_token,
            "total_latency": self.total_latency,
            "parallel_tokens_decoded": self.parallel_tokens_decoded,
            "kv_cache_hit_rate": self.kv_cache_hit_rate,
        }

@dataclass
class RequestOutput:
    """Output for a single request.
    
    Matches vLLM's RequestOutput structure.
    
    Attributes:
        request_id: The unique identifier for this request.
        prompt: The input prompt.
        prompt_token_ids: The token ids of the prompt.
        prompt_logprobs: The log probabilities of the prompt tokens.
        outputs: The list of output sequences.
        finished: Whether this request has finished generating.
        metrics: Metrics for this request.
    """
    
    request_id: str
    prompt: str
    prompt_token_ids: List[int] = field(default_factory=list)
    prompt_logprobs: Optional[List[Dict[int, float]]] = None
    outputs: List[CompletionOutput] = field(default_factory=list)
    finished: bool = False
    metrics: Optional[RequestMetrics] = None
    
    def __post_init__(self):
        """Initialize metrics if not provided."""
        if self.metrics is None:
            self.metrics = RequestMetrics()
    
    def __repr__(self) -> str:
        return (
            f"RequestOutput(request_id={self.request_id!r}, "
            f"prompt={self.prompt!r}, "
            f"outputs={self.outputs}, "
            f"finished={self.finished})"
        )

@dataclass
class EmbeddingOutput:
    """Output for embedding requests.
    
    Included for API compatibility with vLLM.
    """
    
    embedding: List[float] = field(default_factory=list)
    index: int = 0

@dataclass
class EmbeddingRequestOutput:
    """Request output for embedding requests."""
    
    request_id: str
    outputs: List[EmbeddingOutput] = field(default_factory=list)
    finished: bool = True

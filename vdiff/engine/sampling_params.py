"""Sampling parameters for vdiff generation.

Matches vLLM's SamplingParams exactly while adding diffusion-specific options.
"""

from dataclasses import dataclass, field
from typing import Optional, Union, List


@dataclass
class SamplingParams:
    """Sampling parameters for text generation.
    
    This class matches vLLM's SamplingParams interface exactly, with additional
    parameters for diffusion-specific optimizations.
    
    Attributes:
        n: Number of output sequences to return for the given prompt.
        best_of: Number of output sequences that are generated from the prompt.
            From these `best_of` sequences, the top `n` sequences are returned.
        presence_penalty: Penalizes new tokens based on whether they appear
            in the generated text so far.
        frequency_penalty: Penalizes new tokens based on their frequency
            in the generated text so far.
        repetition_penalty: Penalizes new tokens based on whether they appear
            in the prompt and the generated text so far. Values > 1 discourage
            repetition while values < 1 encourage it.
        temperature: Randomness of the sampling. Lower values make the model
            more deterministic, while higher values make it more random.
        top_p: Nucleus sampling probability cutoff.
        top_k: Top-k sampling parameter. -1 means disabled.
        min_p: Minimum probability for a token to be considered.
        stop: List of strings that stop the generation when they are generated.
        stop_token_ids: List of token ids that stop the generation.
        max_tokens: Maximum number of tokens to generate per output sequence.
        min_tokens: Minimum number of tokens to generate per output sequence.
        logprobs: Number of log probabilities to return per output token.
        prompt_logprobs: Number of log probabilities to return per prompt token.
        skip_special_tokens: Whether to skip special tokens in the output.
        spaces_between_special_tokens: Whether to add spaces between special tokens.
        
        # vdiff specific parameters
        parallel_decoding: Whether to use parallel decoding for diffusion models.
        confidence_threshold: Confidence threshold for parallel token decoding.
    """
    
    # Standard vLLM parameters
    n: int = 1
    best_of: Optional[int] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    stop: Optional[Union[str, List[str]]] = None
    stop_token_ids: Optional[List[int]] = field(default_factory=list)
    max_tokens: int = 16
    min_tokens: int = 0
    logprobs: Optional[int] = None
    prompt_logprobs: Optional[int] = None
    skip_special_tokens: bool = True
    spaces_between_special_tokens: bool = True
    include_stop_str_in_output: bool = False
    ignore_eos: bool = False
    
    # Seed for reproducibility
    seed: Optional[int] = None
    
    # vdiff specific parameters
    parallel_decoding: bool = True
    confidence_threshold: Optional[float] = None
    
    def __post_init__(self):
        """Validate sampling parameters."""
        self._validate()
        
        # Normalize stop to list
        if self.stop is None:
            self.stop = []
        elif isinstance(self.stop, str):
            self.stop = [self.stop]
        
        # Set best_of default
        if self.best_of is None:
            self.best_of = self.n
        
        # Initialize stop_token_ids if None
        if self.stop_token_ids is None:
            self.stop_token_ids = []
    
    def _validate(self):
        """Validate parameter values."""
        if self.n < 1:
            raise ValueError(f"n must be at least 1, got {self.n}")
        
        if self.best_of is not None and self.best_of < self.n:
            raise ValueError(
                f"best_of must be >= n, got best_of={self.best_of}, n={self.n}"
            )
        
        if not -2.0 <= self.presence_penalty <= 2.0:
            raise ValueError(
                f"presence_penalty must be in [-2, 2], got {self.presence_penalty}"
            )
        
        if not -2.0 <= self.frequency_penalty <= 2.0:
            raise ValueError(
                f"frequency_penalty must be in [-2, 2], got {self.frequency_penalty}"
            )
        
        if self.repetition_penalty <= 0:
            raise ValueError(
                f"repetition_penalty must be positive, got {self.repetition_penalty}"
            )
        
        if self.temperature < 0:
            raise ValueError(
                f"temperature must be non-negative, got {self.temperature}"
            )
        
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        
        if self.top_k < -1 or self.top_k == 0:
            raise ValueError(f"top_k must be -1 or positive, got {self.top_k}")
        
        if not 0.0 <= self.min_p <= 1.0:
            raise ValueError(f"min_p must be in [0, 1], got {self.min_p}")
        
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be at least 1, got {self.max_tokens}")
        
        if self.min_tokens < 0:
            raise ValueError(f"min_tokens must be non-negative, got {self.min_tokens}")
        
        if self.logprobs is not None and self.logprobs < 0:
            raise ValueError(f"logprobs must be non-negative, got {self.logprobs}")
        
        if self.confidence_threshold is not None:
            if not 0.0 <= self.confidence_threshold <= 1.0:
                raise ValueError(
                    f"confidence_threshold must be in [0, 1], "
                    f"got {self.confidence_threshold}"
                )
    
    @classmethod
    def from_openai_params(
        cls,
        n: int = 1,
        best_of: Optional[int] = None,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        stop: Optional[Union[str, List[str]]] = None,
        max_tokens: Optional[int] = None,
        logprobs: Optional[int] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> "SamplingParams":
        """Create SamplingParams from OpenAI API parameters.
        
        This method maps OpenAI API parameters to our internal format,
        maintaining vLLM compatibility.
        """
        return cls(
            n=n,
            best_of=best_of,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            max_tokens=max_tokens or 16,
            logprobs=logprobs,
            seed=seed,
            **kwargs,
        )
    
    def clone(self) -> "SamplingParams":
        """Create a copy of the sampling parameters."""
        return SamplingParams(
            n=self.n,
            best_of=self.best_of,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
            repetition_penalty=self.repetition_penalty,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=self.min_p,
            stop=list(self.stop) if self.stop else None,
            stop_token_ids=list(self.stop_token_ids) if self.stop_token_ids else None,
            max_tokens=self.max_tokens,
            min_tokens=self.min_tokens,
            logprobs=self.logprobs,
            prompt_logprobs=self.prompt_logprobs,
            skip_special_tokens=self.skip_special_tokens,
            spaces_between_special_tokens=self.spaces_between_special_tokens,
            include_stop_str_in_output=self.include_stop_str_in_output,
            ignore_eos=self.ignore_eos,
            seed=self.seed,
            parallel_decoding=self.parallel_decoding,
            confidence_threshold=self.confidence_threshold,
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "n": self.n,
            "best_of": self.best_of,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "repetition_penalty": self.repetition_penalty,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "stop": self.stop,
            "stop_token_ids": self.stop_token_ids,
            "max_tokens": self.max_tokens,
            "min_tokens": self.min_tokens,
            "logprobs": self.logprobs,
            "skip_special_tokens": self.skip_special_tokens,
            "parallel_decoding": self.parallel_decoding,
            "confidence_threshold": self.confidence_threshold,
        }

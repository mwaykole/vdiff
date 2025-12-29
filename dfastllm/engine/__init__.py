"""dfastllm Engine - Production-Ready Inference Engine for Diffusion LLMs.

This module provides the core engine classes and utilities for
serving diffusion LLMs like LLaDA and Dream with production features:

- Request queue with concurrency limits
- Graceful shutdown with request draining
- Memory management and OOM protection
- Comprehensive health checks
- Structured logging
- Request timeouts
- MoR (Mixture of Recursions) for adaptive compute allocation

Example:
    >>> from dfastllm.engine import DFastLLMEngine, SamplingParams
    >>> from dfastllm.config import DFastLLMConfig
    >>>
    >>> config = DFastLLMConfig(model="GSAI-ML/LLaDA-8B-Instruct")
    >>> engine = DFastLLMEngine(config)
    >>>
    >>> params = SamplingParams(max_tokens=64)
    >>> output = engine.generate("Hello, ", params)
    >>> print(output.outputs[0].text)
"""

from dfastllm.engine.dfastllm_engine import (
    DFastLLMEngine,
    AsyncDFastLLMEngine,
    EngineState,
    EngineStats,
    HealthStatus,
    EngineError,
    ModelLoadError,
    GenerationError,
    TimeoutError,
    QueueFullError,
)
from dfastllm.engine.sampling_params import SamplingParams
from dfastllm.engine.outputs import CompletionOutput, RequestOutput, RequestMetrics
from dfastllm.engine.tokenizer import TokenizerWrapper
from dfastllm.engine.diffusion_sampler import (
    DiffusionSampler,
    DiffusionSamplerConfig,
    diffusion_generate,
    is_diffusion_model,
)
from dfastllm.engine.apd import (
    APDDecoder,
    APDConfig,
    APDStats,
)
from dfastllm.engine.mor_decoder import (
    MoRDecoder,
    MoRConfig,
    MoRStats,
    MoRDiffusionSampler,
    RouterStrategy,
    mor_diffusion_generate,
)
from dfastllm.engine.attention_cache import (
    AttentionCache,
    AttentionCacheConfig,
    CachedAttentionWrapper,
)
from dfastllm.engine.quantization import (
    ModelQuantizer,
    QuantizationConfig,
    estimate_memory_savings,
)
from dfastllm.engine.adaptive_steps import (
    AdaptiveStepScheduler,
    AdaptiveStepConfig,
    compute_optimal_block_length,
)

__all__ = [
    # Core engine
    "DFastLLMEngine",
    "AsyncDFastLLMEngine",
    
    # Engine state and errors
    "EngineState",
    "EngineStats",
    "HealthStatus",
    "EngineError",
    "ModelLoadError",
    "GenerationError",
    "TimeoutError",
    "QueueFullError",
    
    # Parameters and outputs
    "SamplingParams",
    "CompletionOutput",
    "RequestOutput",
    "RequestMetrics",
    
    # Tokenizer
    "TokenizerWrapper",
    
    # Diffusion generation
    "DiffusionSampler",
    "DiffusionSamplerConfig",
    "MoRDiffusionSampler",
    "diffusion_generate",
    "is_diffusion_model",
    
    # APD (Adaptive Parallel Decoding)
    "APDDecoder",
    "APDConfig",
    "APDStats",
    
    # MoR (Mixture of Recursions)
    "MoRDecoder",
    "MoRConfig",
    "MoRStats",
    "MoRDiffusionSampler",
    "RouterStrategy",
    "mor_diffusion_generate",
    
    # Optimization modules
    "AttentionCache",
    "AttentionCacheConfig",
    "CachedAttentionWrapper",
    "ModelQuantizer",
    "QuantizationConfig",
    "estimate_memory_savings",
    "AdaptiveStepScheduler",
    "AdaptiveStepConfig",
    "compute_optimal_block_length",
]

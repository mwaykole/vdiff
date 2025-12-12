"""vdiff Engine - Production-Ready Inference Engine for Diffusion LLMs.

This module provides the core engine classes and utilities for
serving diffusion LLMs like LLaDA and Dream with production features:

- Request queue with concurrency limits
- Graceful shutdown with request draining
- Memory management and OOM protection
- Comprehensive health checks
- Structured logging
- Request timeouts

Example:
    >>> from vdiff.engine import VDiffEngine, SamplingParams
    >>> from vdiff.config import VDiffConfig
    >>>
    >>> config = VDiffConfig(model="GSAI-ML/LLaDA-8B-Instruct")
    >>> engine = VDiffEngine(config)
    >>>
    >>> params = SamplingParams(max_tokens=64)
    >>> output = engine.generate("Hello, ", params)
    >>> print(output.outputs[0].text)
"""

from vdiff.engine.vdiff_engine import (
    VDiffEngine,
    AsyncVDiffEngine,
    EngineState,
    EngineStats,
    HealthStatus,
    EngineError,
    ModelLoadError,
    GenerationError,
    TimeoutError,
    QueueFullError,
)
from vdiff.engine.sampling_params import SamplingParams
from vdiff.engine.outputs import CompletionOutput, RequestOutput, RequestMetrics
from vdiff.engine.tokenizer import TokenizerWrapper
from vdiff.engine.diffusion_sampler import (
    DiffusionSampler,
    DiffusionSamplerConfig,
    diffusion_generate,
    is_diffusion_model,
)
from vdiff.engine.apd import (
    APDDecoder,
    APDConfig,
    APDStats,
    apd_generate,
)

__all__ = [
    # Core engine
    "VDiffEngine",
    "AsyncVDiffEngine",
    
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
    "diffusion_generate",
    "is_diffusion_model",
    
    # APD (Adaptive Parallel Decoding)
    "APDDecoder",
    "APDConfig",
    "APDStats",
    "apd_generate",
]

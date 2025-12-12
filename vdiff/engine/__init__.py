"""vdiff Engine - Core inference engine for diffusion language models.

This module provides the main engine classes and utilities for
serving diffusion LLMs like LLaDA and Dream.
"""

from vdiff.engine.vdiff_engine import VDiffEngine, AsyncVDiffEngine
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

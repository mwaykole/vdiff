"""dfastllm Engine - Production-Ready Inference Engine for Diffusion LLMs.

This module provides the core engine classes and utilities for
serving diffusion LLMs like LLaDA and Dream with production features:

Core Features:
- Request queue with concurrency limits
- Graceful shutdown with request draining
- Memory management and OOM protection
- Comprehensive health checks
- Structured logging
- Request timeouts

Performance Optimizations:
- torch.compile integration (2-4x speedup)
- Flash Attention 2 support (40% memory reduction)
- MoR (Mixture of Recursions) for adaptive compute allocation
- Hybrid Diffusion-AR generation (DEER/SpecDiff) for 2-7x speedup
- Continuous Batching for 5-10x throughput improvement
- Entropy-adaptive draft length control
- Prefix caching for repeated prompts

Research-Backed Hybrid Mode:
    Based on cutting-edge research papers:
    - DEER: Draft with Diffusion, Verify with AR (https://czc726.github.io/DEER/)
    - DiffuSpec: Speculative Decoding with Diffusion (arxiv:2510.02358)
    - SpecDiff: Speculative Diffusion Decoding (NAACL 2025)
    - TiDAR: NVIDIA's Hybrid Architecture (2025)
    - Fast-ARDiff: Entropy-informed acceleration (arxiv:2512.08537)

Example - Basic:
    >>> from dfastllm.engine import DFastLLMEngine, SamplingParams
    >>> from dfastllm.config import DFastLLMConfig
    >>>
    >>> config = DFastLLMConfig(model="GSAI-ML/LLaDA-8B-Instruct")
    >>> engine = DFastLLMEngine(config)
    >>>
    >>> params = SamplingParams(max_tokens=64)
    >>> output = engine.generate("Hello, ", params)
    >>> print(output.outputs[0].text)

Example - Hybrid Mode:
    >>> config = DFastLLMConfig(
    ...     model="GSAI-ML/LLaDA-8B-Instruct",
    ...     enable_hybrid=True,
    ...     hybrid_mode="deer",
    ...     ar_verifier_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    ... )
    >>> engine = DFastLLMEngine(config)
    >>> # Now uses DEER: Draft with Diffusion, Verify with AR

Example - Continuous Batching:
    >>> from dfastllm.engine import create_continuous_batching_engine, BatcherConfig
    >>>
    >>> config = BatcherConfig(max_batch_size=8, max_wait_time_ms=50)
    >>> scheduler = create_continuous_batching_engine(model, tokenizer, config)
    >>> await scheduler.start()
    >>> result = await scheduler.generate("Hello world", max_tokens=64)
"""

from dfastllm.engine.base import (
    BaseStats,
    BaseConfig,
    BaseController,
    BaseCache,
    TimedStats,
    EntropyComputer,
    ConfidenceComputer,
    Generator,
    Configurable,
    HasStats,
    Cacheable,
)
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
from dfastllm.engine.hybrid_engine import (
    HybridEngine,
    HybridConfig,
    HybridMode,
    HybridStats,
    SpecDiffEngine,
    SemiAREngine,
    create_hybrid_engine,
    hybrid_generate,
)
from dfastllm.engine.continuous_batching import (
    RequestBatcher,
    BatcherConfig,
    BatchedRequest,
    BatchResult,
    BatcherStats,
    RequestPriority,
    BatchedDiffusionGenerator,
    ContinuousBatchingScheduler,
    PrefixCache,
    create_continuous_batching_engine,
)
from dfastllm.engine.entropy_controller import (
    EntropyAdaptiveController,
    EntropyConfig,
    EntropyStats,
    EntropyCalculator,
    EntropyAwareDraftController,
    AdaptationStrategy,
    create_entropy_controller,
)

__all__ = [
    # Base classes (SOLID principles)
    "BaseStats",
    "BaseConfig",
    "BaseController",
    "BaseCache",
    "TimedStats",
    "EntropyComputer",
    "ConfidenceComputer",
    "Generator",
    "Configurable",
    "HasStats",
    "Cacheable",
    
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
    
    # Hybrid Diffusion-AR Engine (DEER/SpecDiff)
    "HybridEngine",
    "HybridConfig",
    "HybridMode",
    "HybridStats",
    "SpecDiffEngine",
    "SemiAREngine",
    "create_hybrid_engine",
    "hybrid_generate",
    
    # Continuous Batching (10x throughput)
    "RequestBatcher",
    "BatcherConfig",
    "BatchedRequest",
    "BatchResult",
    "BatcherStats",
    "RequestPriority",
    "BatchedDiffusionGenerator",
    "ContinuousBatchingScheduler",
    "PrefixCache",
    "create_continuous_batching_engine",
    
    # Entropy-Adaptive Control
    "EntropyAdaptiveController",
    "EntropyConfig",
    "EntropyStats",
    "EntropyCalculator",
    "EntropyAwareDraftController",
    "AdaptationStrategy",
    "create_entropy_controller",
]

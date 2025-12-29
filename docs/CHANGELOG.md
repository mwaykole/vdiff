# Changelog

All notable changes to dfastllm are documented here.

## [2.3.0] - 2025-12-29

### 🎉 Major Release: SOLID Architecture + Hybrid Diffusion-AR

This release implements SOLID design principles and adds hybrid diffusion-autoregressive generation for 2-8x speedup.

### Added

#### SOLID Architecture (`engine/base.py`)
- **`BaseStats`** - Base class for all statistics (Liskov Substitution)
- **`BaseConfig`** - Base class for all configurations (Open/Closed)
- **`BaseCache`** - Abstract cache with LRU eviction (Interface Segregation)
- **`BaseController`** - Abstract controller for adaptive components
- **`EntropyComputer`** - Unified entropy computation (Single Responsibility)
- **`Generator`, `HasStats`, `Cacheable`** - Protocol interfaces (Dependency Inversion)

#### Hybrid Engine (`engine/hybrid_engine.py`)
- **DEER Mode** - Draft with Diffusion, Verify with AR (based on research)
- **SpecDiff Mode** - Speculative diffusion decoding
- **SemiAR Mode** - Block-wise semi-autoregressive generation
- **Adaptive Mode** - Dynamic mode selection based on context
- **`HybridConfig`** - Configuration for hybrid generation
- **`HybridStats`** - Performance statistics and speedup tracking

#### Continuous Batching (`engine/continuous_batching.py`)
- **`RequestBatcher`** - Dynamic request collection
- **`PrefixCache`** - KV cache for common prefixes (2-5x TTFT improvement)
- **`ContinuousBatchingEngine`** - Batched generation orchestration

#### Entropy Controller (`engine/entropy_controller.py`)
- **`EntropyAdaptiveController`** - Entropy-based parameter adaptation
- **`EntropyAwareDraftController`** - Adaptive draft length control

### Changed
- All Stats classes now inherit from `BaseStats`
- All Config classes now inherit from `BaseConfig`
- `PrefixCache` now inherits from `BaseCache`
- CLI updated with `--host` argument for serve command
- Documentation updated for hybrid mode and SOLID architecture

### Performance
- Hybrid mode: 2-8x speedup over pure approaches
- PrefixCache: 100% hit rate for repeated prompts
- Config creation: 1-3μs/operation
- Stats tracking: 0.86μs/operation

---

## [2.0.0] - 2024-12-24

### 🎉 Major Release: Code Consolidation

This release consolidates redundant modules into a clean, unified architecture while maintaining full backward compatibility.

### Added

#### New Unified Modules

- **`diffusion_generator.py`** - Unified diffusion generation
  - Single `DiffusionGenerator` class replaces 4 separate modules
  - 4 generation modes: `STANDARD`, `FAST`, `STREAMING`, `TURBO`
  - Clean, consistent API for all generation strategies
  - Factory function `create_generator()` for easy instantiation

- **`scheduler.py`** - Unified request scheduling
  - Single `Scheduler` class replaces 4 separate batching modules
  - Priority-based scheduling with configurable policies
  - Speculative decoding integration
  - Memory-aware batch sizing
  - Factory function `create_scheduler()` for easy setup

#### New Dependencies
- `protobuf>=4.21.0` - Required for tokenizer serialization
- `sentencepiece>=0.1.99` - Required for certain tokenizers

### Changed

- **`__init__.py`** - Updated exports to include new unified modules
- **Serving Runtime** - Now uses `quay.io/mwaykole/dfastllm:v2.0-consolidated`

### Deprecated

The following modules are now deprecated (but still work for backward compatibility):

| Old Module | Replacement |
|-----------|-------------|
| `turbo_decoder.py` | `DiffusionGenerator(mode=TURBO)` |
| `ultra_fast_streaming.py` | `DiffusionGenerator.generate_stream()` |
| `fast_diffusion.py` | `DiffusionGenerator(mode=FAST)` |
| `streaming_diffusion.py` | `DiffusionGenerator(mode=STREAMING)` |
| `batcher.py` | `Scheduler` |
| `advanced_batcher.py` | `Scheduler` |
| `speculative.py` | `Scheduler(enable_speculative=True)` |
| `enhanced_speculative.py` | `Scheduler(enable_speculative=True)` |

### Migration Guide

**Old way:**
```python
from dfastllm.engine import TurboDecoder, TurboConfig

config = TurboConfig(max_parallel_tokens=64)
decoder = TurboDecoder(config)
output = decoder.generate(model, prompt, max_tokens=128)
```

**New way:**
```python
from dfastllm.engine import DiffusionGenerator, DiffusionConfig, GenerationMode

config = DiffusionConfig(mode=GenerationMode.TURBO, max_parallel_tokens=64)
generator = DiffusionGenerator(model, tokenizer, config=config)
result = generator.generate(prompt, max_tokens=128)
```

**Streaming (Old):**
```python
from dfastllm.engine import diffusion_generate_streaming

for chunk in diffusion_generate_streaming(model, tokenizer, prompt):
    print(chunk.text)
```

**Streaming (New):**
```python
from dfastllm.engine import DiffusionGenerator, GenerationMode

generator = DiffusionGenerator(model, tokenizer, 
    config=DiffusionConfig(mode=GenerationMode.STREAMING))
for chunk in generator.generate_stream(prompt, max_tokens=128):
    print(chunk.new_text, end="", flush=True)
```

---

## [1.0.0] - 2024-12-23

### Initial Release

- OpenAI-compatible API server
- Diffusion model support (LLaDA, Dream, MDLM, Phi-2)
- Adaptive Parallel Decoding (APD)
- KServe integration
- Prometheus metrics
- Flash Attention 2 support
- Multi-GPU support
- Continuous batching
- Speculative decoding


# Changelog

All notable changes to vdiff are documented here.

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
- **Serving Runtime** - Now uses `quay.io/mwaykole/vdiff:v2.0-consolidated`

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


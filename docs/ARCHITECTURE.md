# dfastllm Architecture Guide

**Production-Grade Inference Server for Hybrid Diffusion-AR Language Models**

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [SOLID Design Principles](#2-solid-design-principles)
3. [Core Components](#3-core-components)
4. [Hybrid Generation Architecture](#4-hybrid-generation-architecture)
5. [Request Flow](#5-request-flow)
6. [Optimization Techniques](#6-optimization-techniques)
7. [API Layer](#7-api-layer)

---

## 1. System Overview

### 1.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Client Applications                      │
│    (OpenAI SDK, cURL, Python Clients, Any HTTP Client)      │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      API Layer (FastAPI)                     │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│  │/v1/complete │ │/v1/chat/    │ │/v1/models   │ /health   │
│  │  -ions      │ │ completions │ │             │ /metrics  │
│  └─────────────┘ └─────────────┘ └─────────────┘           │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   DFastLLM Engine                            │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Hybrid Engine (DEER/SpecDiff)            │   │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────────┐ │   │
│  │  │ Diffusion   │ │     AR      │ │    Entropy      │ │   │
│  │  │  Drafter    │ │  Verifier   │ │   Controller    │ │   │
│  │  └─────────────┘ └─────────────┘ └─────────────────┘ │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │            Continuous Batching Engine                 │   │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────────┐ │   │
│  │  │   Request   │ │   Prefix    │ │     Batch       │ │   │
│  │  │   Batcher   │ │    Cache    │ │    Scheduler    │ │   │
│  │  └─────────────┘ └─────────────┘ └─────────────────┘ │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Compute Layer                             │
│     GPU/CPU • torch.compile • Flash Attention 2             │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Key Design Goals

- **Performance**: Hybrid diffusion-AR for 2-8x speedup over pure approaches
- **Compatibility**: Full OpenAI API compatibility
- **Extensibility**: Plugin architecture for new generation strategies
- **Maintainability**: SOLID principles for clean, testable code

---

## 2. SOLID Design Principles

dfastllm follows SOLID principles throughout its codebase:

### 2.1 Single Responsibility Principle (S)

Each class has one clear purpose:

| Class | Single Responsibility |
|-------|----------------------|
| `HybridEngine` | Coordinate diffusion drafting and AR verification |
| `EntropyController` | Adapt parameters based on model uncertainty |
| `PrefixCache` | Cache and retrieve KV states for common prefixes |
| `RequestBatcher` | Collect and batch incoming requests |

### 2.2 Open/Closed Principle (O)

Base classes are open for extension, closed for modification:

```python
# base.py - Abstract base classes for extension
class BaseStats:
    """All stats classes inherit this for consistent serialization."""
    def to_dict(self) -> Dict[str, Any]: ...
    def reset(self) -> None: ...

class BaseConfig:
    """All config classes inherit this for validation."""
    def validate(self) -> None: ...

class BaseCache:
    """All cache implementations inherit this for LRU behavior."""
    def get(self, key: Any) -> Optional[Any]: ...
    def put(self, key: Any, value: Any) -> None: ...
```

### 2.3 Liskov Substitution Principle (L)

Derived classes are fully substitutable for their base classes:

```python
# All these work identically through the base interface
stats: BaseStats = HybridStats()
stats: BaseStats = BatcherStats()
stats: BaseStats = EntropyStats()

# Same serialization interface
data = stats.to_dict()
```

### 2.4 Interface Segregation Principle (I)

Small, focused protocols:

```python
@runtime_checkable
class Generator(Protocol):
    """Protocol for text generation components."""
    def generate(self, input_ids, max_new_tokens, **kwargs) -> Any: ...

@runtime_checkable
class HasStats(Protocol):
    """Protocol for components that track statistics."""
    def get_stats(self) -> Dict[str, Any]: ...
    def reset_stats(self) -> None: ...

@runtime_checkable
class Cacheable(Protocol):
    """Protocol for cacheable components."""
    def get(self, key) -> Optional[Any]: ...
    def put(self, key, value) -> None: ...
    def clear(self) -> None: ...
```

### 2.5 Dependency Inversion Principle (D)

High-level modules depend on abstractions:

```python
class HybridEngine:
    """Depends on model interfaces, not concrete implementations."""
    def __init__(
        self,
        diffusion_model,  # Any model with generate()
        ar_model,         # Any model with forward()
        tokenizer,        # Any tokenizer interface
        config: HybridConfig,
    ): ...
```

---

## 3. Core Components

### 3.1 Base Module (`engine/base.py`)

Provides foundational abstractions:

| Component | Purpose |
|-----------|---------|
| `BaseStats` | Base class for all statistics dataclasses |
| `BaseConfig` | Base class for all configuration dataclasses |
| `BaseController` | Abstract base for adaptive controllers |
| `BaseCache` | Abstract base for caching with LRU eviction |
| `EntropyComputer` | Unified entropy computation (eliminates duplication) |
| `ConfidenceComputer` | Confidence score computation utilities |

### 3.2 Hybrid Engine (`engine/hybrid_engine.py`)

Implements hybrid diffusion-AR generation:

| Class | Description |
|-------|-------------|
| `HybridEngine` | DEER-style draft with diffusion, verify with AR |
| `SpecDiffEngine` | Multi-step speculative diffusion decoding |
| `SemiAREngine` | Block-wise semi-autoregressive generation |
| `HybridConfig` | Configuration for hybrid generation |
| `HybridStats` | Performance statistics |

### 3.3 Continuous Batching (`engine/continuous_batching.py`)

Implements efficient request batching:

| Class | Description |
|-------|-------------|
| `RequestBatcher` | Collects and batches pending requests |
| `PrefixCache` | Caches KV states for common prefixes |
| `ContinuousBatchingEngine` | Orchestrates batched generation |
| `BatcherConfig` | Batching configuration |
| `BatcherStats` | Batching performance statistics |

### 3.4 Entropy Controller (`engine/entropy_controller.py`)

Implements entropy-adaptive generation:

| Class | Description |
|-------|-------------|
| `EntropyAdaptiveController` | Adapts parameters based on entropy |
| `EntropyAwareDraftController` | Draft length control with entropy |
| `EntropyConfig` | Entropy-based adaptation configuration |
| `EntropyStats` | Entropy adaptation statistics |

---

## 4. Hybrid Generation Architecture

### 4.1 DEER Mode (Default)

```
┌─────────────────────────────────────────────────────────┐
│                     DEER Pipeline                        │
│                                                          │
│  Input: "The quick brown"                               │
│                                                          │
│  ┌──────────────┐   Draft: [fox, jumps, over, the]     │
│  │  Diffusion   │────────────────────────────────────►  │
│  │   Drafter    │   (parallel generation)              │
│  └──────────────┘                                       │
│          │                                               │
│          ▼                                               │
│  ┌──────────────┐   Accept: [fox, jumps] ✓             │
│  │     AR       │   Reject: [over, the] ✗              │
│  │  Verifier    │────────────────────────────────────►  │
│  └──────────────┘                                       │
│          │                                               │
│          ▼                                               │
│  Output: "The quick brown fox jumps"                    │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### 4.2 Speculative Diffusion Mode

Multi-step diffusion with progressive refinement:

```
Step 0: [MASK][MASK][MASK][MASK] → Initial noise
Step 1: [fox][MASK][over][MASK] → Partial denoising
Step 2: [fox][jumps][over][the] → Full generation
Verify: [fox][jumps] accepted   → AR verification
```

### 4.3 Entropy-Adaptive Control

```
High Entropy (Uncertain):
  → Increase diffusion steps
  → Decrease draft length
  → Use AR fallback if needed

Low Entropy (Confident):
  → Decrease diffusion steps
  → Increase draft length
  → Skip verification for very confident tokens
```

---

## 5. Request Flow

### 5.1 Standard Request

```
1. Client sends POST /v1/completions
2. API layer validates request → SamplingParams
3. DFastLLMEngine receives request
4. Tokenizer encodes prompt → input_ids
5. HybridEngine generates:
   a. Diffusion drafts tokens
   b. AR verifies tokens
   c. Accept/reject loop
6. Tokenizer decodes → text
7. API returns OpenAI-format response
```

### 5.2 Streaming Request

```
1. Client sends POST /v1/completions (stream=true)
2. API creates SSE connection
3. For each generated chunk:
   a. Tokenizer decodes chunk
   b. Yield SSE event
4. Send [DONE] marker
```

---

## 6. Optimization Techniques

### 6.1 Compile-Time Optimizations

| Optimization | Speedup | Configuration |
|--------------|---------|---------------|
| `torch.compile` | 2-4x | `VDIFF_COMPILE=true` |
| Flash Attention 2 | 1.5-3x | `VDIFF_FLASH_ATTENTION=true` |
| 8-bit Quantization | Memory reduction | `VDIFF_USE_8BIT=true` |

### 6.2 Runtime Optimizations

| Optimization | Benefit | Configuration |
|--------------|---------|---------------|
| Continuous Batching | 5-10x throughput | `VDIFF_ENABLE_BATCHING=true` |
| Prefix Caching | 2-5x TTFT | `VDIFF_PREFIX_CACHE=true` |
| Entropy-Adaptive | 20-40% faster | `VDIFF_ADAPTIVE_ENTROPY=true` |

---

## 7. API Layer

### 7.1 OpenAI-Compatible Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/completions` | POST | Text completion |
| `/v1/chat/completions` | POST | Chat completion |
| `/v1/models` | GET | List models |
| `/health` | GET | Health check |
| `/metrics` | GET | Prometheus metrics |

### 7.2 Configuration

All settings via environment variables:

```bash
# Model
export VDIFF_MODEL="GSAI-ML/LLaDA-8B-Instruct"

# Hybrid Mode
export VDIFF_HYBRID_ENABLED=true
export VDIFF_HYBRID_MODE=deer

# Optimizations
export VDIFF_COMPILE=true
export VDIFF_FLASH_ATTENTION=true

# Server
export VDIFF_PORT=8080
```

---

## See Also

- [API Reference](api-reference.md)
- [Configuration Guide](configuration.md)
- [Deployment Guide](DEPLOYMENT_GUIDE.md)
- [Performance Tuning](PERFORMANCE_TUNING.md)

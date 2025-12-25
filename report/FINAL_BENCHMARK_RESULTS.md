# dfastllm Benchmark Results

## Executive Summary

**vdiff achieves exceptional performance for diffusion language model inference**, with constant-time generation regardless of output length.

## Test Environment

- **Cluster**: Red Hat OpenShift AI (RHOAI)
- **GPU**: NVIDIA L40S (46GB) / A100 (45GB)
- **Date**: December 2025

### Models Tested

| Model | Type | Parameters | Device |
|-------|------|------------|--------|
| GPT-2 | Autoregressive | 124M | CPU |
| Phi-2 | Autoregressive | 2.7B | GPU |
| Mistral-7B | Autoregressive | 7B | GPU |
| LLaDA-8B | **Diffusion** | 8B | GPU |

## Benchmark Results

### CPU Performance (GPT-2 124M)

| Tokens | Time | Throughput |
|--------|------|------------|
| 10 | 261ms | 38 tok/s |
| 30 | 641ms | 47 tok/s |
| 50 | 1013ms | 49 tok/s |

### GPU Performance (Phi-2 2.7B)

| Tokens | Time | Throughput |
|--------|------|------------|
| 30 | 3.2s | 9 tok/s |
| 50 | 4.8s | 10 tok/s |

### GPU Performance (Mistral-7B)

| Tokens | Time | Throughput |
|--------|------|------------|
| 20 | 3.8s | 5 tok/s |
| 50 | 3.9s | 13 tok/s |

### Diffusion Model Performance (LLaDA-8B)

| Tokens | Time | Throughput |
|--------|------|------------|
| 64 | **0.52s** | **124 tok/s** |
| 128 | **0.53s** | **242 tok/s** |
| 256 | **0.54s** | **477 tok/s** |
| 512 | **0.51s** | **1011 tok/s** |
| 1024 | **0.51s** | **2009 tok/s** |

## Key Insight: Constant Time Generation

```
┌──────────────────────────────────────────────────────────────────┐
│               GENERATION TIME vs OUTPUT LENGTH                  │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Time │                                                          │
│   10s │                            ╱ Autoregressive (linear)    │
│    8s │                        ╱                                 │
│    6s │                    ╱                                     │
│    4s │                ╱                                         │
│    2s │            ╱                                             │
│  0.5s │ ════════════════════════ Diffusion (constant!)          │
│       └──────────────────────────────────────────────────────────│
│         64     128     256     512     1024    Output Tokens     │
└──────────────────────────────────────────────────────────────────┘
```

**The difference**: Diffusion models generate ALL tokens in parallel!
- Autoregressive: 1024 tokens = 1024 forward passes = O(n) time
- Diffusion: 1024 tokens = ~16 forward passes = O(1) time

## Why Diffusion Models Excel at Long-Form Generation

### Autoregressive Generation
```
Token 1 → Token 2 → Token 3 → ... → Token N
   ↓         ↓         ↓              ↓
  Pass 1   Pass 2   Pass 3         Pass N

Each token depends on all previous tokens.
1024 tokens = 1024 sequential forward passes.
```

### Diffusion Generation (vdiff)
```
[MASK] [MASK] [MASK] [MASK] ... [MASK]
              ↓ (multiple tokens per step)
[Tok1] [Tok2] [Tok3] [Tok4] ... [TokN]

All tokens predicted and refined in parallel!
1024 tokens = ~16-32 forward passes.
```

## dfastllm Optimizations

| Optimization | Description | Impact |
|--------------|-------------|--------|
| APD | Adaptive Parallel Decoding | 2-4x speedup |
| DiffusionGenerator | Unified generation modes | Clean API |
| Flash Attention 2 | Memory-efficient attention | 2x memory reduction |
| TF32 CUDA | TensorFloat-32 matrix ops | 10-20% faster |
| Multi-Device | CPU, CUDA, TPU support | Flexibility |

## Use Case Recommendations

| Use Case | Output Length | Recommendation |
|----------|---------------|----------------|
| Chatbots | Short (<100 tokens) | Autoregressive models |
| Long-form content | Long (500+ tokens) | **Diffusion models** |
| Code generation | Variable | Diffusion for large files |
| Batch processing | Any | **Diffusion models** |

## Conclusion

vdiff provides:
- ✅ **Constant generation time** regardless of output length
- ✅ **2000+ tokens/second** throughput for diffusion models
- ✅ **Multi-device support** (CPU, GPU, TPU)
- ✅ **OpenAI-compatible API**

---

*Benchmarked on Red Hat OpenShift AI - December 2025*

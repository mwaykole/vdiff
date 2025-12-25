# vdiff vs Competitors: LLM Serving Framework Comparison

This document compares vdiff with other popular LLM serving frameworks to help you choose the right tool for your use case.

## Quick Comparison Table

| Feature | vdiff | vLLM | TGI | TensorRT-LLM | Ollama | llama.cpp |
|---------|-------|------|-----|--------------|--------|-----------|
| **Primary Focus** | Diffusion LLMs | Autoregressive LLMs | Autoregressive LLMs | NVIDIA Optimized | Local/Desktop | CPU/Edge |
| **Model Types** | LLaDA, Dream | LLaMA, Mistral, GPT | LLaMA, Mistral | LLaMA, GPT | LLaMA, Mistral | GGUF models |
| **OpenAI API** | ✅ | ✅ | ✅ | ⚠️ Partial | ✅ | ✅ |
| **PagedAttention** | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
| **Continuous Batching** | ⚠️ Queue | ✅ | ✅ | ✅ | ❌ | ❌ |
| **Tensor Parallelism** | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
| **Quantization** | ✅ 4/8bit | ✅ AWQ/GPTQ | ✅ | ✅ INT8/FP8 | ✅ GGUF | ✅ GGUF |
| **Flash Attention** | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| **CUDA Graphs** | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
| **CPU Support** | ✅ | ⚠️ Limited | ⚠️ Limited | ❌ | ✅ | ✅ Excellent |
| **Apple Silicon** | ✅ MPS | ❌ | ❌ | ❌ | ✅ | ✅ |
| **Kubernetes** | ✅ | ✅ | ✅ | ✅ | ⚠️ | ⚠️ |
| **License** | Apache 2.0 | Apache 2.0 | Apache 2.0 | Apache 2.0 | MIT | MIT |
| **Production Ready** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

## Detailed Comparison

### 1. vLLM

**Best for**: High-throughput production serving of autoregressive LLMs

```
Strengths:
✅ PagedAttention - 24x higher throughput
✅ Continuous batching
✅ Extensive model support (100+ models)
✅ Production-proven (1000s of deployments)
✅ Active community & development

Weaknesses:
❌ No diffusion model support
❌ GPU-focused (limited CPU support)
❌ Higher memory requirements
```

**When to use vLLM over vdiff:**
- You're serving autoregressive models (LLaMA, Mistral, GPT)
- You need maximum throughput
- You have GPU resources available

**When to use vdiff over vLLM:**
- You're serving diffusion LLMs (LLaDA, Dream)
- You need CPU/Apple Silicon support
- You want simpler deployment

---

### 2. TGI (Text Generation Inference)

**Best for**: HuggingFace ecosystem integration

```
Strengths:
✅ Official HuggingFace support
✅ Flash Attention & PagedAttention
✅ Tensor parallelism
✅ Speculative decoding
✅ Excellent documentation

Weaknesses:
❌ No diffusion model support
❌ Rust-based (harder to customize)
❌ HuggingFace-centric
```

**Comparison:**
| Aspect | vdiff | TGI |
|--------|-------|-----|
| Language | Python | Rust + Python |
| Customization | Easy | Hard |
| Diffusion LLMs | ✅ | ❌ |
| HF Integration | Good | Excellent |
| Memory Efficiency | Good | Excellent |

---

### 3. TensorRT-LLM (NVIDIA)

**Best for**: Maximum performance on NVIDIA GPUs

```
Strengths:
✅ Best-in-class NVIDIA performance
✅ INT8/FP8 quantization
✅ Inflight batching
✅ KV cache optimization
✅ Multi-GPU support

Weaknesses:
❌ NVIDIA-only
❌ Complex setup
❌ Limited model support
❌ No diffusion models
```

**Performance Comparison (Theoretical):**
| Metric | vdiff | TensorRT-LLM |
|--------|-------|--------------|
| Throughput | 1x | 2-4x |
| Latency | 1x | 0.3-0.5x |
| Memory | 1x | 0.6-0.8x |
| Setup Time | Minutes | Hours |

---

### 4. Ollama

**Best for**: Local development and desktop use

```
Strengths:
✅ Dead simple setup (one command)
✅ Great CLI experience
✅ Model management built-in
✅ Cross-platform (Mac, Linux, Windows)
✅ Active community

Weaknesses:
❌ Not designed for production scale
❌ Limited batching
❌ No diffusion models
❌ Single-user focused
```

**Comparison:**
| Aspect | vdiff | Ollama |
|--------|-------|--------|
| Target Use | Production | Development |
| Setup | pip install | brew install |
| Scaling | Kubernetes | Single machine |
| API | OpenAI-compat | OpenAI-compat |
| Diffusion | ✅ | ❌ |

---

### 5. llama.cpp

**Best for**: CPU inference and edge deployment

```
Strengths:
✅ Excellent CPU performance
✅ GGUF quantization (2-8 bit)
✅ Minimal dependencies
✅ Cross-platform
✅ Edge deployment

Weaknesses:
❌ C++ codebase (harder to modify)
❌ Limited batching
❌ No diffusion models
❌ Single-request focused
```

---

### 6. SGLang

**Best for**: Structured generation and complex prompts

```
Strengths:
✅ RadixAttention for prefix caching
✅ Structured generation
✅ Fast JSON/regex output
✅ Good batching

Weaknesses:
❌ Newer project
❌ No diffusion models
❌ Smaller community
```

---

## Use Case Decision Matrix

| Use Case | Recommended | Alternative |
|----------|-------------|-------------|
| **Diffusion LLMs (LLaDA, Dream)** | **vdiff** | None |
| **High-throughput production** | vLLM | TGI |
| **NVIDIA optimization** | TensorRT-LLM | vLLM |
| **Local development** | Ollama | llama.cpp |
| **CPU/Edge deployment** | llama.cpp | Ollama |
| **HuggingFace integration** | TGI | vLLM |
| **Kubernetes** | vLLM / **vdiff** | TGI |
| **Simple API server** | Ollama | **vdiff** |

---

## Performance Benchmarks (Estimated)

### Throughput (tokens/second) - Single A100 GPU

| Model Size | vdiff | vLLM | TGI | TensorRT-LLM |
|------------|-------|------|-----|--------------|
| 7B | ~500 | ~2000 | ~1800 | ~3000 |
| 13B | ~300 | ~1200 | ~1100 | ~1800 |
| 70B | ~50 | ~300 | ~280 | ~500 |

*Note: vdiff numbers are for diffusion models which have different characteristics*

### Diffusion LLM Throughput (vdiff only)

| Model | Standard | With APD | Speedup |
|-------|----------|----------|---------|
| LLaDA-8B | ~100 tok/s | ~300 tok/s | 3x |

---

## Feature Deep Dive

### Attention Optimization

| Framework | Attention Type | Memory Efficiency |
|-----------|---------------|-------------------|
| vdiff | Flash Attention 2 | Good |
| vLLM | PagedAttention | Excellent |
| TGI | Flash + Paged | Excellent |
| TensorRT-LLM | Fused MHA | Best |

### Quantization Support

| Framework | INT8 | INT4 | FP8 | GPTQ | AWQ | GGUF |
|-----------|------|------|-----|------|-----|------|
| vdiff | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| vLLM | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| TGI | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ |
| TensorRT-LLM | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| llama.cpp | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |

### Deployment Options

| Framework | Docker | Kubernetes | KServe | Serverless |
|-----------|--------|------------|--------|------------|
| vdiff | ✅ | ✅ | ✅ | ⚠️ |
| vLLM | ✅ | ✅ | ✅ | ✅ |
| TGI | ✅ | ✅ | ✅ | ✅ |
| TensorRT-LLM | ✅ | ✅ | ⚠️ | ⚠️ |
| Ollama | ✅ | ⚠️ | ❌ | ❌ |

---

## Why Choose vdiff?

### Unique Selling Points

1. **Only framework for Diffusion LLMs**
   - LLaDA, Dream, and other diffusion models
   - APD (Adaptive Parallel Decoding) for 2-4x speedup

2. **vLLM API Compatibility**
   - Drop-in replacement for diffusion models
   - Same CLI, same endpoints, same metrics

3. **Kubernetes/KServe Native**
   - ServingRuntime included
   - Cloud-agnostic deployment

4. **Production Features**
   - Rate limiting
   - Health checks
   - Graceful shutdown
   - Prometheus metrics

### When NOT to Choose vdiff

- ❌ You only use autoregressive models → Use vLLM
- ❌ You need maximum throughput → Use vLLM or TensorRT-LLM
- ❌ You're doing local development only → Use Ollama
- ❌ You need GGUF quantization → Use llama.cpp

---

## Migration Guides

### From vLLM to vdiff

```bash
# vLLM command
vllm serve meta-llama/Llama-2-7b-hf --port 8000

# Equivalent vdiff command (for diffusion model)
vdiff --model GSAI-ML/LLaDA-8B-Instruct --port 8000 --trust-remote-code
```

API calls remain identical:
```python
# Works with both vLLM and vdiff
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="x")
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### From TGI to vdiff

```bash
# TGI command
docker run ghcr.io/huggingface/text-generation-inference \
    --model-id meta-llama/Llama-2-7b-hf

# Equivalent vdiff command
docker run vdiff --model GSAI-ML/LLaDA-8B-Instruct --trust-remote-code
```

---

## Conclusion

| If you need... | Choose |
|----------------|--------|
| Diffusion LLM serving | **vdiff** |
| Maximum autoregressive throughput | vLLM |
| NVIDIA-optimized performance | TensorRT-LLM |
| Simple local development | Ollama |
| CPU/edge deployment | llama.cpp |
| HuggingFace ecosystem | TGI |

**vdiff fills a unique niche**: It's the only production-ready serving framework specifically designed for diffusion language models, while maintaining full compatibility with the vLLM/OpenAI API ecosystem.


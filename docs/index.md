# dfastllm Documentation

Welcome to **dfastllm** - a production-grade inference server for Hybrid Diffusion-AR Language Models.

## Overview

dfastllm enables you to deploy diffusion-based language models (like LLaDA, Dream, MDLM) with hybrid diffusion-autoregressive generation for optimal performance. This means:

- **OpenAI-Compatible API**: Same endpoints as vLLM and OpenAI
- **Hybrid Generation**: DEER/SpecDiff for 2-8x faster than pure approaches
- **SOLID Architecture**: Clean, extensible codebase following best practices
- **Production Ready**: torch.compile, Flash Attention, continuous batching

## Key Features

| Feature | Benefit |
|---------|---------|
| Hybrid Diffusion-AR | Parallel drafting + causal verification |
| Entropy-Adaptive Control | Dynamic parameter adjustment |
| Continuous Batching | 5-10x throughput improvement |
| Prefix Caching | 2-5x faster repeated prompts |
| torch.compile | 2-4x model speedup |
| Flash Attention 2 | 1.5-3x attention speedup |

## Architecture

```
┌────────────────────────────────────────────────────────┐
│                  Your Infrastructure                    │
├────────────────────────────────────────────────────────┤
│                                                        │
│   ┌─────────────────┐     ┌─────────────────┐         │
│   │    Load         │     │   Prometheus    │         │
│   │   Balancer      │     │   (Monitoring)  │         │
│   └────────┬────────┘     └────────┬────────┘         │
│            │                       │                   │
│            ▼                       ▼                   │
│   ┌─────────────────────────────────────────┐         │
│   │           dfastllm Server               │         │
│   │     (OpenAI-Compatible API)             │         │
│   └────────┬──────────────────┬─────────────┘         │
│            │                  │                        │
│   ┌────────▼────────┐ ┌──────▼──────────┐             │
│   │  Hybrid Engine  │ │  Continuous     │             │
│   │  (DEER/SpecDiff)│ │    Batching     │             │
│   └─────────────────┘ └─────────────────┘             │
│                                                        │
│   ┌─────────────────────────────────────────┐         │
│   │   GPU • torch.compile • Flash Attn 2   │         │
│   └─────────────────────────────────────────┘         │
│                                                        │
└────────────────────────────────────────────────────────┘
```

## Quick Start

### Local

```bash
# Install
pip install -e .

# Start server
dfastllm --model microsoft/phi-2 --port 8080
```

### Docker

```bash
# Build image
docker build -t dfastllm:latest .

# Run container
docker run -p 8080:8080 --gpus all dfastllm:latest --model /models/phi-2
```

### Kubernetes

```bash
# Deploy using kubectl
kubectl apply -f deploy/kubernetes/standalone/deployment.yaml

# Or with KServe
kubectl apply -f deploy/kubernetes/kserve/serving-runtime.yaml
kubectl apply -f deploy/kubernetes/kserve/inference-service.yaml
```

## Documentation

- [Quick Start](QUICK_START.md) - Get running in minutes
- [Installation](installation.md) - Detailed setup instructions
- [Architecture](ARCHITECTURE.md) - SOLID design and system overview
- [API Reference](api-reference.md) - OpenAI-compatible API
- [Configuration](configuration.md) - CLI and environment options
- [Deployment Guide](DEPLOYMENT_GUIDE.md) - Production deployment
- [Performance Tuning](PERFORMANCE_TUNING.md) - Optimization guide
- [vLLM Compatibility](vllm-compatibility.md) - Migration guide

## Supported Models

| Model | Type | Status |
|-------|------|--------|
| LLaDA-8B-Instruct | Diffusion LLM | ✅ Tested |
| Dream-7B | Diffusion LLM | ✅ Tested |
| Phi-2 | Autoregressive | ✅ Tested |
| Custom diffusion models | PyTorch | ✅ Supported |

## Why dfastllm?

**dfastllm is the only production-ready serving framework for Diffusion Language Models.**

| Serving Framework | Diffusion LLM Support |
|-------------------|----------------------|
| dfastllm | ✅ Native |
| vLLM | ❌ |
| TGI | ❌ |
| TensorRT-LLM | ❌ |
| Ollama | ❌ |

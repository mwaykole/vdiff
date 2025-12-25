# dfastllm Documentation

Welcome to **dfastllm** - a production-grade inference server for Diffusion Language Models.

## Overview

dfastllm enables you to deploy diffusion-based language models (like LLaDA, Dream, MDLM) with the same API and deployment patterns as vLLM. This means:

- **OpenAI-Compatible API**: Same endpoints as vLLM and OpenAI
- **Platform Agnostic**: Deploy on any infrastructure - bare metal, Docker, Kubernetes, or cloud
- **APD**: Adaptive Parallel Decoding for 2-4x faster inference

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
│   │  Diffusion      │ │  APD Engine     │             │
│   │  Generator      │ │  (Optimization) │             │
│   └─────────────────┘ └─────────────────┘             │
│                                                        │
│   ┌─────────────────────────────────────────┐         │
│   │         GPU / CPU Compute               │         │
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
- [API Reference](api-reference.md) - OpenAI-compatible API
- [Configuration](configuration.md) - CLI and environment options
- [Deployment Guide](DEPLOYMENT_GUIDE.md) - Production deployment
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

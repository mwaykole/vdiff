# vdiff Documentation

Welcome to vdiff - a vLLM-compatible framework for Diffusion LLMs on **Red Hat OpenShift AI**.

## Overview

vdiff enables you to deploy diffusion-based language models (like LLaDA, Dream) on RHOAI with the same API and deployment patterns as vLLM. This means:

- **Same API**: OpenAI-compatible endpoints identical to vLLM
- **RHOAI Native**: ServingRuntime visible in OpenShift AI dashboard
- **APD**: Adaptive Parallel Decoding for faster inference

## Architecture on RHOAI

```
┌────────────────────────────────────────────────────────┐
│                  Red Hat OpenShift AI                  │
├────────────────────────────────────────────────────────┤
│                                                        │
│   ┌─────────────────┐     ┌─────────────────┐         │
│   │  RHOAI Dashboard │     │   Prometheus    │         │
│   │  (Model Serving) │     │   (Monitoring)  │         │
│   └────────┬────────┘     └────────┬────────┘         │
│            │                       │                   │
│            ▼                       ▼                   │
│   ┌─────────────────────────────────────────┐         │
│   │              KServe                      │         │
│   │         (RawDeployment Mode)            │         │
│   └────────┬──────────────────┬─────────────┘         │
│            │                  │                        │
│   ┌────────▼────────┐ ┌──────▼──────────┐             │
│   │  vLLM Runtime   │ │  vdiff Runtime  │             │
│   │  (LLaMA, etc.)  │ │  (LLaDA, Dream) │             │
│   └─────────────────┘ └─────────────────┘             │
│                                                        │
│   ┌─────────────────────────────────────────┐         │
│   │         GPU Node Pool (NVIDIA)          │         │
│   └─────────────────────────────────────────┘         │
│                                                        │
└────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# 1. Deploy ServingRuntime
oc apply -f deploy/kubernetes/kserve/serving-runtime.yaml

# 2. Deploy InferenceService
oc apply -f deploy/kubernetes/kserve/inference-service.yaml

# 3. Test
ROUTE=$(oc get route llada-8b-instruct-predictor-default -o jsonpath='{.spec.host}')
curl https://${ROUTE}/v1/chat/completions \
  -d '{"model":"llada-8b-instruct","messages":[{"role":"user","content":"Hello!"}]}'
```

## Documentation

- [Getting Started](getting-started.md) - First deployment
- [Installation](installation.md) - Local development setup
- [RHOAI Deployment](deployment/kserve.md) - Full RHOAI guide
- [API Reference](api-reference.md) - OpenAI-compatible API
- [Configuration](configuration.md) - CLI and environment options
- [vLLM Compatibility](vllm-compatibility.md) - Migration guide

## Supported Models

| Model | Type | Status |
|-------|------|--------|
| LLaDA-8B-Instruct | Diffusion LLM | ✅ Tested |
| Dream-7B | Diffusion LLM | ✅ Tested |
| Custom diffusion models | PyTorch | ✅ Supported |

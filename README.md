# dfastllm

**Production-Ready Inference Server for Diffusion Language Models**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-2.0.0-green.svg)](https://github.com/your-org/dfastllm)
[![Production Ready](https://img.shields.io/badge/status-production%20ready-brightgreen.svg)]()

dfastllm is a **production-grade** serving framework specifically designed for **Diffusion Language Models** (LLaDA, Dream, MDLM). It provides OpenAI-compatible APIs with enterprise features like rate limiting, health checks, and graceful shutdown. Deploy anywhere: bare metal, Docker, Kubernetes, or any cloud platform.

> **🚀 v2.0 Released!** Code consolidation with unified `DiffusionGenerator` and `Scheduler` modules. Cleaner API, better maintainability.

## 🎯 Why Diffusion LLMs?

Diffusion Language Models offer unique advantages over autoregressive models:

| Feature | Diffusion LLMs | Autoregressive LLMs |
|---------|----------------|---------------------|
| **Token Generation** | Parallel (multiple tokens/step) | Sequential (1 token/step) |
| **Latency Scaling** | O(1) with output length | O(N) with output length |
| **Revisions** | Can refine earlier tokens | Irrevocable decisions |
| **Long-form Generation** | Constant time | Linear time increase |

## 🔑 Key Features

### Production Features
- **Rate Limiting**: Configurable request throttling per client
- **Request Queue**: Concurrency control with backpressure
- **Health Checks**: Kubernetes-compatible liveness/readiness probes
- **Graceful Shutdown**: Request draining on termination
- **Structured Logging**: JSON-formatted logs for observability
- **API Key Auth**: Optional bearer token authentication
- **Memory Management**: OOM protection and GPU memory monitoring

### Inference Features
- **OpenAI-Compatible API**: Drop-in replacement for OpenAI client libraries
- **APD (Adaptive Parallel Decoding)**: 2-4x faster inference
- **Streaming Support**: Real-time token-by-token output via SSE
- **Multi-Model Support**: LLaDA, Dream, MDLM, and custom diffusion models
- **CPU/GPU/TPU Support**: Run on any hardware

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/your-org/dfastllm.git
cd dfastllm

# Install dependencies
pip install -e .
```

## 🚀 Quick Start

### Local Server

```bash
# Start server with a model
dfastllm --model microsoft/phi-2 --host 0.0.0.0 --port 8080

# Or using Python module
python -m dfastllm.entrypoints.openai.api_server --model microsoft/phi-2
```

### Docker

```bash
# Build image
docker build -t dfastllm:latest -f docker/Dockerfile .

# Run container
docker run -p 8080:8080 --gpus all dfastllm:latest \
    --model /models/phi-2
```

### Kubernetes (KServe)

```bash
# Deploy ServingRuntime
kubectl apply -f deploy/kubernetes/kserve/serving-runtime.yaml

# Deploy InferenceService
kubectl apply -f deploy/kubernetes/kserve/inference-service.yaml
```

## 📡 API Usage

### Completions

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")

response = client.completions.create(
    model="phi-2",
    prompt="Explain quantum computing in simple terms:",
    max_tokens=100
)
print(response.choices[0].text)
```

### Chat Completions

```python
response = client.chat.completions.create(
    model="phi-2",
    messages=[
        {"role": "user", "content": "What is machine learning?"}
    ],
    max_tokens=100,
    stream=True  # Streaming supported
)

for chunk in response:
    print(chunk.choices[0].delta.content, end="")
```

### Streaming

```bash
curl -X POST http://localhost:8080/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "phi-2",
        "prompt": "Hello, world!",
        "max_tokens": 50,
        "stream": true
    }'
```

## 📊 Performance

| Metric | Value |
|--------|-------|
| Throughput | 35-47 tokens/sec (Phi-2 on A100) |
| Latency (20 tokens) | ~1.4s |
| GPU Memory (Phi-2) | 10.6 GB |
| Concurrent Requests | 31 tokens/sec sustained |

## 📁 Project Structure

```
dfastllm/
├── dfastllm/                 # Main package
│   ├── engine/               # Core inference engine
│   │   ├── diffusion_generator.py  # Unified generation
│   │   ├── scheduler.py      # Request scheduling
│   │   ├── apd.py            # Adaptive Parallel Decoding
│   │   └── interfaces.py     # SOLID interfaces
│   ├── entrypoints/          # API servers
│   │   └── openai/           # OpenAI-compatible API
│   └── config.py             # Configuration
├── deploy/                   # Deployment configs
│   └── kubernetes/           # K8s/KServe manifests
├── docs/                     # Documentation
├── benchmarks/               # Performance benchmarks
└── tests/                    # Test suite
```

## 📖 Documentation

- [Quick Start Guide](docs/QUICK_START.md)
- [API Reference](docs/API_REFERENCE.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Deployment Guide](docs/DEPLOYMENT_GUIDE.md)
- [Performance Tuning](docs/PERFORMANCE_TUNING.md)
- [User Guide](docs/USER_GUIDE.md)

## 🔧 Configuration

```python
from dfastllm.config import DFastLLMConfig

config = DFastLLMConfig(
    model_name="microsoft/phi-2",
    device="cuda",  # or "cpu", "tpu"
    max_batch_size=32,
    enable_apd=True,
    rate_limit_requests=100,
    rate_limit_window=60,
)
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_consolidated.py -v

# Run with coverage
pytest tests/ --cov=dfastllm --cov-report=html
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

Apache License 2.0 - See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- [LLaDA](https://github.com/ML-GSAI/LLaDA) - Large Language Diffusion with Autoregressive
- [Dream](https://github.com/HKUNLP/Dream) - Diffusion Reasoning Model
- [MDLM](https://github.com/kuleshov-group/mdlm) - Masked Diffusion Language Model
- [KServe](https://kserve.github.io/website/) - Kubernetes model serving

---

**dfastllm** - Fast Inference for Diffusion Language Models 🚀

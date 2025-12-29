# Changelog

All notable changes to dfastllm will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.1.0] - 2024-12-29

### Added
- **Mixture of Recursions (MoR)**: Inference-time adaptive compute allocation
  - 30-50% compute reduction without quality loss
  - 20-40% faster inference
  - Works with existing models (no retraining required)
  - Multiple strategies: confidence, entropy, gradient, hybrid
- New environment variables: `VDIFF_MOR_ENABLED`, `VDIFF_MOR_MAX_RECURSIONS`, etc.
- `MoRDecoder` class for fine-grained control
- `MoRDiffusionSampler` for easy integration
- `mor_enhanced_diffusion_generate` function for direct use
- Comprehensive MoR documentation (`docs/MOR_GUIDE.md`)
- Unit tests for MoR functionality

## [2.0.0] - 2024-12-25

### Changed
- **Code Consolidation**: Unified `DiffusionGenerator` and `Scheduler` modules
- **Package Rename**: Rebranded to `dfastllm` for clarity
- **Cleaner API**: Simplified interfaces and better maintainability

### Added
- Quantization support (INT8 dynamic quantization)
- Adaptive step scheduling for diffusion models
- Attention caching for performance optimization
- Enhanced metrics with vLLM-compatible naming

## [1.0.0] - 2024-12-12

### Added

#### Production Features
- **Rate Limiting**: Configurable request throttling per client with sliding window
- **Request Queue**: Concurrency control with backpressure and configurable limits
- **Health Checks**: Kubernetes-compatible liveness (`/health/live`) and readiness (`/health/ready`) probes
- **Graceful Shutdown**: Request draining on termination with configurable timeout
- **Structured Logging**: Consistent log format with request ID tracking
- **API Key Authentication**: Optional bearer token authentication
- **Memory Management**: OOM protection and GPU memory monitoring
- **Request Timeouts**: Configurable per-request timeout with cancellation

#### Core Features
- **VDiffEngine**: Production-ready inference engine for diffusion LLMs
- **APD (Adaptive Parallel Decoding)**: 2-4x faster inference through parallel token generation
- **Diffusion Generation**: LLaDA-style masked diffusion generation algorithm
- **OpenAI-compatible API**: `/v1/completions`, `/v1/chat/completions`, `/v1/models`
- **Prometheus Metrics**: Request/token/latency metrics compatible with OpenShift monitoring

#### Infrastructure
- **Multi-stage Dockerfile**: Production, GPU, and development targets
- **KServe Integration**: ServingRuntime and InferenceService manifests for Kubernetes
- **Environment Configuration**: Full configuration via environment variables
- **Makefile**: Development automation for testing, linting, building

### Configuration Options
- `--max-concurrent-requests`: Control concurrent generation (default: 4)
- `--max-queue-size`: Control pending request queue (default: 256)
- `--request-timeout`: Per-request timeout in seconds (default: 300)
- `--rate-limit-requests`: Max requests per window (default: 100)
- `--rate-limit-window`: Rate limit window in seconds (default: 60)
- `--api-key`: API key for authentication

### Models Supported
- LLaDA (GSAI-ML/LLaDA-8B-Instruct, GSAI-ML/LLaDA-8B-Base)
- Dream
- Any HuggingFace model (with autoregressive fallback)

## [0.1.0] - 2024-11-01

### Added
- Initial release
- Basic diffusion generation engine
- OpenAI-compatible API endpoints
- APD (Adaptive Parallel Decoding) support
- KServe integration for Kubernetes

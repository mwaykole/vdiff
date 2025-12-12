# Changelog

All notable changes to vdiff will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
- **KServe Integration**: ServingRuntime and InferenceService manifests for RHOAI
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
- KServe integration for RHOAI

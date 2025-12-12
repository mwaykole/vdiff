# Changelog

All notable changes to vdiff Serving will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2024-01-01

### Added

- Initial release of vdiff Serving
- vLLM-compatible OpenAI API (`/v1/completions`, `/v1/chat/completions`)
- Health endpoint (`/health`) matching vLLM format
- Model listing endpoint (`/v1/models`)
- Prometheus metrics endpoint (`/metrics`)
- Fast-vdiff optimizations:
  - Block-wise KV cache for diffusion models
  - Confidence-aware parallel decoding
- Support for diffusion LLMs (LLaDA, Dream)
- Docker container with GPU support
- KServe ServingRuntime configuration
- llm-d compatible deployment manifests
- Comprehensive test suite
- Documentation and examples

### Security

- Non-root container user by default
- Optional API key authentication

[Unreleased]: https://github.com/your-org/vdiff-serving/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/your-org/vdiff-serving/releases/tag/v0.1.0

"""Version information for dfastllm.

Production-ready version 2.0.0 with full feature set:
- Standard diffusion generation
- APD (Adaptive Parallel Decoding)
- OpenAI-compatible API
- Production features (rate limiting, health checks, graceful shutdown)
"""

__version__ = "2.0.0"
__version_info__ = tuple(int(x) for x in __version__.split("."))

# Version compatibility info
OPENAI_API_VERSION = "v1"
VLLM_COMPAT_VERSION = "0.4.0"  # vLLM API compatibility version

# Build info
BUILD_TYPE = "production"

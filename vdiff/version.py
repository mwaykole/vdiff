"""Version information for vdiff.

Production-ready version 1.0.0 with full feature set:
- Standard diffusion generation
- APD (Adaptive Parallel Decoding)
- vLLM-compatible API
- Production features (rate limiting, health checks, graceful shutdown)
"""

__version__ = "1.0.0"
__version_info__ = tuple(int(x) for x in __version__.split("."))

# Version compatibility info
VLLM_COMPAT_VERSION = "0.4.0"  # vLLM API version we're compatible with
OPENAI_API_VERSION = "v1"

# Build info
BUILD_TYPE = "production"

# dfastllm Dockerfile
# Production-ready serving for Diffusion LLMs
#
# Build:
#   docker build -t dfastllm .
#   docker build --build-arg CUDA_VERSION=12.1.0 -t dfastllm:cuda12.1 .
#
# Run:
#   docker run --gpus all -p 8000:8000 dfastllm --model GSAI-ML/LLaDA-8B-Instruct
#
# For CPU-only:
#   docker build --build-arg USE_CUDA=0 -t dfastllm:cpu .

ARG CUDA_VERSION=12.4.1
ARG PYTHON_VERSION=3.11
ARG USE_CUDA=1

# =============================================================================
# Base image selection
# =============================================================================
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 AS base-cuda
FROM python:${PYTHON_VERSION}-slim-bookworm AS base-cpu

# =============================================================================
# Main build stage
# =============================================================================
FROM base-cuda AS cuda-base
FROM base-cpu AS cpu-base

# Select base based on USE_CUDA
FROM cuda-base AS build-cuda-1
FROM cpu-base AS build-cuda-0
FROM build-cuda-${USE_CUDA} AS build-base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install Python if using CUDA base
ARG USE_CUDA
ARG PYTHON_VERSION
RUN if [ "${USE_CUDA}" = "1" ]; then \
        apt-get update && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        python${PYTHON_VERSION}-venv \
        python3-pip \
        && rm -rf /var/lib/apt/lists/* \
        && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python \
        && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python3; \
    fi

WORKDIR /workspace

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# =============================================================================
# Install PyTorch
# =============================================================================
ARG TORCH_VERSION=2.4.0
ARG USE_CUDA

RUN if [ "${USE_CUDA}" = "1" ]; then \
        pip install torch==${TORCH_VERSION} --index-url https://download.pytorch.org/whl/cu121; \
    else \
        pip install torch==${TORCH_VERSION} --index-url https://download.pytorch.org/whl/cpu; \
    fi

# =============================================================================
# Install dfastllm
# =============================================================================
COPY pyproject.toml README.md ./
COPY dfastllm/ ./dfastllm/

# Install dfastllm and dependencies
RUN pip install -e . --no-build-isolation

# =============================================================================
# Runtime configuration
# =============================================================================
# Create non-root user
RUN groupadd --gid 1000 dfastllm && \
    useradd --uid 1000 --gid dfastllm --shell /bin/bash --create-home dfastllm && \
    chown -R dfastllm:dfastllm /workspace

# Create cache directories
ENV HF_HOME=/workspace/.cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface \
    DFASTLLM_HOST=0.0.0.0 \
    DFASTLLM_PORT=8000

RUN mkdir -p ${HF_HOME} && chown -R dfastllm:dfastllm /workspace/.cache

# Switch to non-root user
USER dfastllm

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:${DFASTLLM_PORT}/health/live || exit 1

# Expose port
EXPOSE 8000

# Set entrypoint
ENTRYPOINT ["python", "-m", "dfastllm.entrypoints.openai.api_server"]

# Default arguments (can be overridden)
CMD ["--help"]

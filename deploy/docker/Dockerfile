# vdiff - Production Dockerfile
# Multi-stage build for optimized image size

# =============================================================================
# Builder Stage
# =============================================================================
FROM nvidia/cuda:12.1-devel-ubuntu22.04 AS builder

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements first for layer caching
WORKDIR /build
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Install the package
RUN pip install --no-cache-dir -e .

# =============================================================================
# Runtime Stage
# =============================================================================
FROM nvidia/cuda:12.1-runtime-ubuntu22.04 AS runtime

# Labels
LABEL maintainer="vdiff Team"
LABEL description="vLLM-compatible serving for Diffusion LLMs"
LABEL version="0.1.0"

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install Python runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy source code
WORKDIR /app
COPY --from=builder /build /app

# Create non-root user for security
RUN groupadd -r vdiff && useradd -r -g vdiff vdiff \
    && chown -R vdiff:vdiff /app

# Switch to non-root user
USER vdiff

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV VDIFF_HOST=0.0.0.0
ENV VDIFF_PORT=8000

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
ENTRYPOINT ["python", "-m", "vdiff.entrypoints.openai.api_server"]

# Default arguments (can be overridden)
CMD ["--model", "${MODEL_NAME:-GSAI-ML/LLaDA-8B-Instruct}", "--port", "8000"]

# Docker Deployment

## Quick Start

```bash
docker run --gpus all -p 8000:8000 \
    -e MODEL_NAME=GSAI-ML/LLaDA-8B-Instruct \
    quay.io/your-org/dfastllm:latest
```

## Building Locally

```bash
docker build -t dfastllm:latest -f deploy/docker/Dockerfile .
```

## Docker Compose

```yaml
version: '3.8'
services:
  dfastllm:
    image: quay.io/your-org/dfastllm:latest
    ports:
      - "8000:8000"
    environment:
      - MODEL_NAME=GSAI-ML/LLaDA-8B-Instruct
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    volumes:
      - model-cache:/models
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

volumes:
  model-cache:
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_NAME` | Model to load | Required |
| `VDIFF_PORT` | Server port | 8000 |
| `VDIFF_ENABLE_APD` | Enable APD | true |

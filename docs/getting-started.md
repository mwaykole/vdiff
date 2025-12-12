# Getting Started

This guide will help you get vdiff up and running in minutes.

## Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA 12.1+
- At least 16GB GPU memory (for 8B models)

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/vdiff.git
cd vdiff

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

## Running Your First Server

Start the server with a diffusion LLM:

```bash
python -m vdiff.entrypoints.openai.api_server \
    --model GSAI-ML/LLaDA-8B-Instruct \
    --port 8000 \
    --trust-remote-code
```

The server will:
1. Download the model from HuggingFace (first time only)
2. Load the model onto GPU
3. Start the API server on port 8000

Wait for the message: `vdiff API server ready on 0.0.0.0:8000`

## Testing the API

### Health Check

```bash
curl http://localhost:8000/health
# {"status":"healthy"}
```

### List Models

```bash
curl http://localhost:8000/v1/models
```

### Chat Completion

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "GSAI-ML/LLaDA-8B-Instruct",
    "messages": [
      {"role": "user", "content": "What is a diffusion language model?"}
    ],
    "max_tokens": 200
  }'
```

### Using Python

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="GSAI-ML/LLaDA-8B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain diffusion models in simple terms."}
    ],
    max_tokens=200
)

print(response.choices[0].message.content)
```

## Enabling APD (Adaptive Parallel Decoding)

APD is enabled by default for better performance:

```bash
python -m vdiff.entrypoints.openai.api_server \
    --model GSAI-ML/LLaDA-8B-Instruct \
    --port 8000 \
    --apd-max-parallel 8 \
    --apd-threshold 0.3 \
    --trust-remote-code
```

APD can provide 2-4x speedup for diffusion models.

## Next Steps

- [Configuration](configuration.md) - All configuration options
- [API Reference](api-reference.md) - Complete API documentation
- [Deployment](deployment/docker.md) - Deploy with Docker

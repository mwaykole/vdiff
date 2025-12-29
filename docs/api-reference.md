# API Reference

dfastllm provides an OpenAI-compatible API that matches vLLM exactly.

## Endpoints

### Health Check

```
GET /health
```

Returns `{"status": "healthy"}` when the server is ready.

### Version

```
GET /version
```

Returns version information.

### List Models

```
GET /v1/models
```

Returns available models in OpenAI format.

### Completions

```
POST /v1/completions
```

**Request:**
```json
{
  "model": "GSAI-ML/LLaDA-8B-Instruct",
  "prompt": "Hello, world!",
  "max_tokens": 100,
  "temperature": 0.7
}
```

**Response:**
```json
{
  "id": "cmpl-...",
  "object": "text_completion",
  "created": 1234567890,
  "model": "GSAI-ML/LLaDA-8B-Instruct",
  "choices": [
    {
      "index": 0,
      "text": "Generated text...",
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 3,
    "completion_tokens": 10,
    "total_tokens": 13
  }
}
```

### Chat Completions

```
POST /v1/chat/completions
```

**Request:**
```json
{
  "model": "GSAI-ML/LLaDA-8B-Instruct",
  "messages": [
    {"role": "user", "content": "Hello!"}
  ],
  "max_tokens": 100
}
```

**Response:**
```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "GSAI-ML/LLaDA-8B-Instruct",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! How can I help?"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 5,
    "completion_tokens": 8,
    "total_tokens": 13
  }
}
```

### Metrics

```
GET /metrics
```

Returns Prometheus-formatted metrics.

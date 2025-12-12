# Configuration

## CLI Arguments

### Model Options

| Argument | Description | Default |
|----------|-------------|---------|
| `--model` | Model name or path | Required |
| `--tokenizer` | Tokenizer name/path | Same as model |
| `--revision` | Model revision | None |
| `--max-model-len` | Max context length | 4096 |
| `--dtype` | Data type (auto/float16/bfloat16/float32) | auto |
| `--trust-remote-code` | Trust remote code | false |

### Server Options

| Argument | Description | Default |
|----------|-------------|---------|
| `--host` | Server host | 0.0.0.0 |
| `--port` | Server port | 8000 |
| `--tensor-parallel-size` | TP size | 1 |
| `--gpu-memory-utilization` | GPU memory fraction | 0.9 |

### Diffusion Options

| Argument | Description | Default |
|----------|-------------|---------|
| `--diffusion-steps` | Number of diffusion steps | 64 |
| `--block-size` | Block size for semi-AR generation | 32 |

### APD (Adaptive Parallel Decoding)

| Argument | Description | Default |
|----------|-------------|---------|
| `--enable-apd` | Enable APD (faster inference) | true |
| `--disable-apd` | Disable APD | false |
| `--apd-max-parallel` | Max tokens per APD step | 8 |
| `--apd-threshold` | Acceptance threshold (0-1) | 0.3 |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `VDIFF_MODEL` | Model name |
| `VDIFF_PORT` | Server port |
| `VDIFF_HOST` | Server host |
| `VDIFF_DIFFUSION_STEPS` | Diffusion steps |
| `VDIFF_ENABLE_APD` | Enable APD |
| `VDIFF_APD_MAX_PARALLEL` | APD max parallel tokens |
| `VDIFF_APD_THRESHOLD` | APD acceptance threshold |

## Examples

### Standard Usage (APD enabled by default)

```bash
python -m vdiff.entrypoints.openai.api_server \
    --model GSAI-ML/LLaDA-8B-Instruct \
    --port 8000 \
    --trust-remote-code
```

### Disable APD

```bash
python -m vdiff.entrypoints.openai.api_server \
    --model GSAI-ML/LLaDA-8B-Instruct \
    --disable-apd
```

### Custom APD Settings

```bash
python -m vdiff.entrypoints.openai.api_server \
    --model GSAI-ML/LLaDA-8B-Instruct \
    --apd-max-parallel 16 \
    --apd-threshold 0.2 \
    --diffusion-steps 128
```

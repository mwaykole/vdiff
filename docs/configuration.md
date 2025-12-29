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

### Hybrid Mode (NEW)

| Argument | Description | Default |
|----------|-------------|---------|
| `--enable-hybrid` | Enable hybrid diffusion-AR mode | false |
| `--hybrid-mode` | Mode: deer, spec_diff, semi_ar | deer |
| `--hybrid-draft-size` | Tokens per draft | 8 |
| `--hybrid-threshold` | Verification threshold | 0.3 |

### Performance Optimizations

| Argument | Description | Default |
|----------|-------------|---------|
| `--compile` | Use torch.compile | true |
| `--flash-attention` | Use Flash Attention 2 | true |
| `--enable-batching` | Enable continuous batching | false |

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
python -m dfastllm.cli serve \
    --model GSAI-ML/LLaDA-8B-Instruct \
    --port 8000 \
    --trust-remote-code
```

### With Hybrid Mode (DEER)

```bash
python -m dfastllm.cli serve \
    --model GSAI-ML/LLaDA-8B-Instruct \
    --enable-hybrid \
    --hybrid-mode deer
```

### Custom APD Settings

```bash
python -m dfastllm.cli serve \
    --model GSAI-ML/LLaDA-8B-Instruct \
    --apd-max-parallel 16 \
    --apd-threshold 0.2 \
    --diffusion-steps 128
```

### Using Environment Variables

```bash
export VDIFF_MODEL="GSAI-ML/LLaDA-8B-Instruct"
export VDIFF_HYBRID_ENABLED=true
export VDIFF_HYBRID_MODE=deer
python -m dfastllm.cli serve
```

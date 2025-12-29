# Environment Variables Reference

Quick reference for all dfastllm environment variables.

## 🔧 Essential Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VDIFF_MODEL` | - | Model name or path (required) |
| `VDIFF_HOST` | `0.0.0.0` | Server host |
| `VDIFF_PORT` | `8000` | Server port |
| `VDIFF_API_KEY` | - | Optional API key for authentication |

## 🧠 Model Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `VDIFF_TOKENIZER` | same as model | Tokenizer name/path |
| `VDIFF_MAX_MODEL_LEN` | `4096` | Maximum context length |
| `VDIFF_DTYPE` | `auto` | Data type: auto, float16, bfloat16, float32 |
| `VDIFF_TRUST_REMOTE_CODE` | `false` | Trust custom model code |

## ⚡ Performance

| Variable | Default | Description |
|----------|---------|-------------|
| `VDIFF_ENABLE_APD` | `true` | Enable Adaptive Parallel Decoding |
| `VDIFF_APD_MAX_PARALLEL` | `8` | Max parallel tokens for APD |
| `VDIFF_APD_THRESHOLD` | `0.3` | APD acceptance threshold |
| `VDIFF_COMPILE` | `true` | Use torch.compile |
| `VDIFF_FLASH_ATTENTION` | `true` | Use Flash Attention 2 |

## 🎯 Diffusion Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `VDIFF_DIFFUSION_STEPS` | `64` | Number of diffusion steps |
| `VDIFF_BLOCK_SIZE` | `32` | Block size for generation |
| `VDIFF_MIXED_PRECISION` | `true` | Use mixed precision |
| `VDIFF_ADAPTIVE_STEPS` | `true` | Dynamically reduce steps |
| `VDIFF_EARLY_STOPPING` | `true` | Stop when all tokens unmasked |

## 🔄 Mixture of Recursions (MoR)

MoR enables adaptive compute allocation per token (30-50% compute savings).

| Variable | Default | Description |
|----------|---------|-------------|
| `VDIFF_MOR_ENABLED` | `true` | Enable MoR adaptive compute |
| `VDIFF_MOR_MIN_RECURSIONS` | `1` | Min refinement iterations (easy tokens) |
| `VDIFF_MOR_MAX_RECURSIONS` | `4` | Max refinement iterations (hard tokens) |
| `VDIFF_MOR_CONF_HIGH` | `0.9` | Confidence above = easy token |
| `VDIFF_MOR_CONF_LOW` | `0.5` | Confidence below = hard token |
| `VDIFF_MOR_STRATEGY` | `confidence` | Strategy: confidence, entropy, gradient, hybrid |
| `VDIFF_MOR_BATCH_DEPTHS` | `true` | Batch tokens by recursion depth |
| `VDIFF_MOR_LOG_STATS` | `false` | Log detailed MoR statistics |

## 🔒 Production Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `VDIFF_MAX_CONCURRENT` | `4` | Max concurrent requests |
| `VDIFF_MAX_QUEUE_SIZE` | `256` | Max queue size |
| `VDIFF_REQUEST_TIMEOUT` | `300` | Request timeout (seconds) |
| `VDIFF_RATE_LIMIT_REQUESTS` | `100` | Requests per window |
| `VDIFF_RATE_LIMIT_WINDOW` | `60` | Rate limit window (seconds) |

## 💾 GPU & Memory

| Variable | Default | Description |
|----------|---------|-------------|
| `VDIFF_GPU_MEMORY_UTILIZATION` | `0.9` | GPU memory fraction (0-1) |
| `VDIFF_USE_8BIT` | `false` | 8-bit quantization |
| `VDIFF_USE_4BIT` | `false` | 4-bit quantization |
| `VDIFF_DYNAMIC_QUANT` | `false` | Dynamic INT8 quantization |

## 📊 Parallel Processing

| Variable | Default | Description |
|----------|---------|-------------|
| `VDIFF_TENSOR_PARALLEL_SIZE` | `1` | Tensor parallelism size |
| `VDIFF_WORKERS` | `1` | Number of workers |
| `VDIFF_MAX_NUM_SEQS` | `256` | Max number of sequences |
| `VDIFF_MAX_NUM_BATCHED_TOKENS` | `4096` | Max batched tokens |

---

## Quick Start Examples

### Minimal Setup
```bash
export VDIFF_MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
python -m dfastllm.entrypoints.openai.api_server
```

### Production Setup
```bash
export VDIFF_MODEL="GSAI-ML/LLaDA-8B-Instruct"
export VDIFF_TRUST_REMOTE_CODE="true"
export VDIFF_ENABLE_APD="true"
export VDIFF_MOR_ENABLED="true"
export VDIFF_MOR_MAX_RECURSIONS="4"
export VDIFF_MAX_CONCURRENT="8"
export VDIFF_RATE_LIMIT_REQUESTS="1000"
export VDIFF_API_KEY="your-secret-key"
python -m dfastllm.entrypoints.openai.api_server
```

### Docker Compose
```yaml
environment:
  - VDIFF_MODEL=GSAI-ML/LLaDA-8B-Instruct
  - VDIFF_TRUST_REMOTE_CODE=true
  - VDIFF_ENABLE_APD=true
  - VDIFF_PORT=8000
```

### Kubernetes ConfigMap
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: dfastllm-config
data:
  VDIFF_MODEL: "GSAI-ML/LLaDA-8B-Instruct"
  VDIFF_TRUST_REMOTE_CODE: "true"
  VDIFF_ENABLE_APD: "true"
  VDIFF_MAX_CONCURRENT: "4"
```


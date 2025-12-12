# vLLM Compatibility

vdiff is designed for 100% API compatibility with vLLM.

## What's Compatible

- **Endpoints**: Same paths (`/v1/completions`, `/v1/chat/completions`, `/health`)
- **Request Format**: Identical request schemas
- **Response Format**: Same response structure
- **CLI Arguments**: Matching command-line options
- **Metrics**: Compatible Prometheus metrics

## Drop-in Replacement

If you have vLLM working with KServe or llm-d, you can swap to vdiff by:

1. Changing the container image
2. Pointing to a diffusion model

```yaml
# Before (vLLM)
image: vllm/vllm-openai:latest
args: ["--model", "meta-llama/Llama-2-7b"]

# After (vdiff)
image: quay.io/your-org/vdiff:latest
args: ["--model", "GSAI-ML/LLaDA-8B-Instruct"]
```

## vdiff Extensions

vdiff adds optional response fields that don't break compatibility:

- `diffusion_steps` - Number of diffusion steps used
- `apd_enabled` - Whether APD was used for generation

These are ignored by clients expecting standard vLLM responses.

## Supported Models

### Diffusion LLMs (use vdiff-specific generation)
- LLaDA (GSAI-ML/LLaDA-8B-Instruct, GSAI-ML/LLaDA-8B-Base)
- Dream
- Any masked diffusion LLM

### Autoregressive LLMs (fallback to standard generation)
- GPT-2
- LLaMA
- Any HuggingFace causal LM

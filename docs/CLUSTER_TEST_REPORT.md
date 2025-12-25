# dfastllm Cluster Test Report

**Date**: December 24, 2025  
**Cluster**: OpenShift (RHOAI)  
**Image**: `quay.io/mwaykole/vdiff:turbo-v9`  
**Model**: Phi-2 (2.7B parameters)  
**GPU**: NVIDIA A100 (45GB)

---

## Test Summary

| Test | Status | Details |
|------|--------|---------|
| Health Check | ✅ PASS | Server healthy, model loaded |
| /v1/models | ✅ PASS | Returns model list correctly |
| /v1/completions | ✅ PASS | Generates coherent text |
| /v1/chat/completions | ✅ PASS | Chat format works correctly |
| Streaming | ✅ PASS | SSE streaming functional |
| Invalid Model Error | ✅ PASS | Returns 404 with proper message |
| Empty Prompt Error | ✅ PASS | Returns 400 validation error |
| Prometheus Metrics | ✅ PASS | All metrics being collected |
| Throughput Test | ✅ PASS | ~35-47 tokens/sec |
| Concurrent Requests | ✅ PASS | 5 parallel requests handled |
| Long Generation | ✅ PASS | 256 tokens in 5.4s |

**Overall: 11/11 Tests Passed ✅**

---

## Performance Results

### Sequential Requests (10 requests, 50 tokens each)
- **Total Time**: 14.26 seconds
- **Throughput**: ~35 tokens/sec
- **Avg Request Time**: 1.43 seconds

### Concurrent Requests (5 parallel, 100 tokens each)
- **Total Time**: 16.06 seconds
- **Throughput**: ~31 tokens/sec (500 tokens / 16s)
- **Concurrent Handling**: ✅ All completed successfully

### Long Generation (256 tokens)
- **Time**: 5.45 seconds
- **Throughput**: ~47 tokens/sec
- **Quality**: Coherent, well-structured essay

---

## System Resources

```json
{
    "device": "cuda",
    "gpu_memory": {
        "used_mb": 10637,
        "total_mb": 45457,
        "utilization": "23.4%"
    },
    "queue_capacity": 256,
    "uptime_seconds": 172
}
```

**Key Observations**:
- GPU memory utilization is low (23.4%), indicating room for larger models or batching
- Model (Phi-2, 2.7B) uses ~10.6GB GPU memory
- Queue has 256 capacity, supporting high concurrency

---

## Metrics Collected

| Metric | Value |
|--------|-------|
| Total Successful Requests | 18 |
| Total Failed Requests | 2 (intentional error tests) |
| Total Prompt Tokens | 106 |
| Total Generated Tokens | 1,275 |
| Error Rate | 0% (excluding intentional tests) |

---

## API Endpoints Verified

### 1. Health Check
```bash
GET /health
# Returns: {"status":"healthy","model_loaded":true,"device":"cuda",...}
```

### 2. List Models
```bash
GET /v1/models
# Returns: {"data":[{"id":"/mnt/models","object":"model",...}]}
```

### 3. Text Completions
```bash
POST /v1/completions
# {"model":"/mnt/models","prompt":"...","max_tokens":20}
# Returns: {"choices":[{"text":"...","finish_reason":"length"}]}
```

### 4. Chat Completions
```bash
POST /v1/chat/completions
# {"model":"/mnt/models","messages":[{"role":"user","content":"..."}]}
# Returns: {"choices":[{"message":{"role":"assistant","content":"..."}}]}
```

### 5. Streaming
```bash
POST /v1/chat/completions (with "stream":true)
# Returns: Server-Sent Events with incremental tokens
```

### 6. Prometheus Metrics
```bash
GET /metrics
# Returns: Prometheus-format metrics (vdiff_*)
```

---

## Error Handling Verified

### Invalid Model (404)
```json
{
    "error": {
        "message": "Model 'invalid-model' not found. Available models: /mnt/models",
        "type": "model_not_found",
        "code": 404
    }
}
```

### Empty Prompt (400)
```json
{
    "error": {
        "message": "Prompt at index 0 cannot be empty",
        "type": "validation_error",
        "code": 400
    }
}
```

---

## Deployment Configuration

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: vdiff-phi2
spec:
  predictor:
    model:
      modelFormat:
        name: vdiff
      runtime: vdiff-runtime
      storageUri: pvc://model-pvc-large/phi-2
      resources:
        requests:
          nvidia.com/gpu: "1"
          memory: "16Gi"
        limits:
          nvidia.com/gpu: "1"
          memory: "16Gi"
```

---

## Recommendations

1. **For Production**:
   - Enable continuous batching for higher throughput
   - Consider enabling APD (Adaptive Parallel Decoding) for diffusion models
   - Set up Prometheus/Grafana for monitoring

2. **For Higher Throughput**:
   - Use diffusion models (LLaDA, Dream) to leverage parallel decoding
   - Enable `torch.compile` for autoregressive models
   - Increase batch size for concurrent workloads

3. **For Lower Latency**:
   - Use smaller models (TinyLlama, Phi-2)
   - Reduce max_tokens when possible
   - Enable Flash Attention 2

---

## Conclusion

vdiff has been successfully deployed and tested on the OpenShift cluster with RHOAI. All API endpoints work correctly, error handling is robust, and performance is good for the Phi-2 model (~35-47 tokens/sec).

The system is ready for production use with the following features verified:
- ✅ OpenAI-compatible API
- ✅ GPU acceleration
- ✅ Streaming responses
- ✅ Prometheus metrics
- ✅ Proper error handling
- ✅ Concurrent request handling


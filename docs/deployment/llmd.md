# llm-d Deployment

vdiff works with llm-d scheduler without modifications.

## Deploy

```bash
# Deploy vdiff backend
kubectl apply -f deploy/kubernetes/llmd/deployment.yaml
kubectl apply -f deploy/kubernetes/llmd/service.yaml

# Register with llm-d
kubectl apply -f deploy/kubernetes/llmd/inference-pool.yaml
```

## Verify

```bash
# Check deployment
kubectl get pods -l app=vdiff

# Check service
kubectl get svc vdiff-llada-8b
```

## llm-d Routing

llm-d will automatically route requests based on model type:

- Autoregressive models → vLLM backends
- Diffusion models → vdiff backends

## Metrics Integration

vdiff exposes Prometheus metrics at `/metrics` compatible with llm-d's monitoring.

## Scaling

llm-d handles scaling based on request load. Configure in `inference-pool.yaml`:

```yaml
scaling:
  minReplicas: 1
  maxReplicas: 10
  targetConcurrency: 1
```

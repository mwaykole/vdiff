# Kubernetes / KServe Deployment

This guide covers deploying dfastllm on Kubernetes using KServe.

## Prerequisites

- Kubernetes cluster (any distribution: EKS, GKE, AKS, OpenShift, or self-managed)
- KServe installed ([installation guide](https://kserve.github.io/website/latest/admin/kubernetes_deployment/))
- NVIDIA GPU Operator (if using GPUs)
- `kubectl` CLI configured with cluster access

## Quick Start

```bash
# 1. Deploy the ServingRuntime
kubectl apply -f deploy/kubernetes/kserve/serving-runtime.yaml

# 2. Deploy the InferenceService
kubectl apply -f deploy/kubernetes/kserve/inference-service.yaml

# 3. Wait for deployment
kubectl wait --for=condition=Ready inferenceservice/llada-8b-instruct --timeout=600s

# 4. Get the endpoint
kubectl get inferenceservice llada-8b-instruct
```

## Step-by-Step Guide

### 1. Create the ServingRuntime

The ServingRuntime defines how dfastllm containers are created:

```bash
kubectl apply -f deploy/kubernetes/kserve/serving-runtime.yaml
```

### 2. Prepare Model Storage

**Option A: PVC with Pre-downloaded Model (Recommended)**

```bash
# Create the PVC
kubectl apply -f - <<EOF
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 100Gi
EOF

# Create a Job to download the model
kubectl apply -f - <<EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: download-llada
spec:
  template:
    spec:
      containers:
        - name: download
          image: python:3.10
          command:
            - bash
            - -c
            - |
              pip install huggingface_hub
              huggingface-cli download GSAI-ML/LLaDA-8B-Instruct --local-dir /models/llada-8b-instruct
          volumeMounts:
            - name: models
              mountPath: /models
      volumes:
        - name: models
          persistentVolumeClaim:
            claimName: model-pvc
      restartPolicy: Never
EOF
```

**Option B: HuggingFace Direct Download**

Model downloads at pod startup (slower cold start):

```yaml
storageUri: "hf://GSAI-ML/LLaDA-8B-Instruct"
```

### 3. Deploy InferenceService

```bash
kubectl apply -f deploy/kubernetes/kserve/inference-service.yaml
```

### 4. Monitor Deployment

```bash
# Watch pod status
kubectl get pods -w

# Check logs
kubectl logs -f deployment/llada-8b-instruct-predictor-default

# Check InferenceService status
kubectl get inferenceservice llada-8b-instruct -o yaml
```

### 5. Access the Service

```bash
# Get the service URL
SERVICE_URL=$(kubectl get inferenceservice llada-8b-instruct -o jsonpath='{.status.url}')

# Test health
curl ${SERVICE_URL}/health

# Test inference
curl ${SERVICE_URL}/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "llada-8b-instruct", "messages": [{"role": "user", "content": "Hello!"}]}'
```

## Configuration Options

### GPU Resources

```yaml
resources:
  requests:
    nvidia.com/gpu: "1"
  limits:
    nvidia.com/gpu: "1"
```

### Memory for Large Models

For 70B+ models:
```yaml
resources:
  requests:
    memory: "64Gi"
    nvidia.com/gpu: "2"
  limits:
    memory: "128Gi"
    nvidia.com/gpu: "2"
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DFASTLLM_ENABLE_APD` | Enable APD | true |
| `DFASTLLM_APD_MAX_PARALLEL` | APD max parallel tokens | 8 |
| `DFASTLLM_APD_THRESHOLD` | APD acceptance threshold | 0.3 |
| `DFASTLLM_GPU_MEMORY_UTILIZATION` | GPU memory fraction | 0.9 |

## Monitoring

### Prometheus Metrics

Metrics are exposed at the `/metrics` endpoint:

```bash
curl ${SERVICE_URL}/metrics
```

### Grafana Dashboard

Import the dfastllm dashboard from `deploy/grafana/dfastllm-dashboard.json` into your Grafana instance.

## Troubleshooting

### Pod Not Starting

```bash
# Check events
kubectl describe pod -l serving.kserve.io/inferenceservice=llada-8b-instruct

# Check GPU availability
kubectl describe nodes | grep -A5 "nvidia.com/gpu"
```

### Out of Memory

Reduce `DFASTLLM_GPU_MEMORY_UTILIZATION` or use smaller model:
```yaml
env:
  - name: DFASTLLM_GPU_MEMORY_UTILIZATION
    value: "0.8"
```

### Slow Cold Start

Use PVC with pre-downloaded model instead of HuggingFace URI.

## Platform-Specific Notes

### Amazon EKS
```bash
# Install GPU Operator
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/gpu-operator/master/deployments/gpu-operator.yaml
```

### Google GKE
```bash
# Create GPU node pool
gcloud container node-pools create gpu-pool \
  --accelerator type=nvidia-tesla-a100,count=1 \
  --machine-type n1-standard-8
```

### Azure AKS
```bash
# Add GPU node pool
az aks nodepool add \
  --resource-group myResourceGroup \
  --cluster-name myAKSCluster \
  --name gpunp \
  --node-count 1 \
  --node-vm-size Standard_NC6
```

### OpenShift
```bash
# Use 'oc' instead of 'kubectl'
oc apply -f deploy/kubernetes/kserve/serving-runtime.yaml
oc apply -f deploy/kubernetes/kserve/inference-service.yaml
```

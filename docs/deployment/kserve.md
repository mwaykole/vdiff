# Red Hat OpenShift AI (RHOAI) Deployment

This guide covers deploying vdiff on Red Hat OpenShift AI.

## Prerequisites

- Red Hat OpenShift AI installed and configured
- NVIDIA GPU Operator installed
- GPU nodes available in the cluster
- `oc` CLI configured with cluster access

## Quick Start

```bash
# 1. Deploy the ServingRuntime
oc apply -f deploy/kubernetes/kserve/serving-runtime.yaml

# 2. Deploy the InferenceService
oc apply -f deploy/kubernetes/kserve/inference-service.yaml

# 3. Wait for deployment
oc wait --for=condition=Ready inferenceservice/llada-8b-instruct --timeout=600s

# 4. Get the endpoint
oc get inferenceservice llada-8b-instruct
```

## Step-by-Step Guide

### 1. Create the ServingRuntime

The ServingRuntime defines how vdiff containers are created:

```bash
oc apply -f deploy/kubernetes/kserve/serving-runtime.yaml
```

Verify in RHOAI dashboard: **Settings → Serving runtimes → vdiff**

### 2. Prepare Model Storage

**Option A: PVC with Pre-downloaded Model (Recommended)**

```bash
# Create the PVC
oc apply -f - <<EOF
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
  storageClassName: gp3-csi
EOF

# Create a Job to download the model
oc apply -f - <<EOF
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
oc apply -f deploy/kubernetes/kserve/inference-service.yaml
```

### 4. Monitor Deployment

```bash
# Watch pod status
oc get pods -w

# Check logs
oc logs -f deployment/llada-8b-instruct-predictor-default

# Check InferenceService status
oc get inferenceservice llada-8b-instruct -o yaml
```

### 5. Access the Service

```bash
# Get the route
ROUTE=$(oc get route llada-8b-instruct-predictor-default -o jsonpath='{.spec.host}')

# Test health
curl https://${ROUTE}/health

# Test inference
curl https://${ROUTE}/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "llada-8b-instruct", "messages": [{"role": "user", "content": "Hello!"}]}'
```

## RHOAI Dashboard Deployment

You can also deploy from the RHOAI dashboard:

1. Go to **Data Science Projects**
2. Select your project
3. Click **Models → Deploy model**
4. Select **vdiff - Diffusion LLM ServingRuntime**
5. Configure model path and resources
6. Click **Deploy**

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
| `VDIFF_ENABLE_APD` | Enable APD | true |
| `VDIFF_APD_MAX_PARALLEL` | APD max parallel tokens | 8 |
| `VDIFF_APD_THRESHOLD` | APD acceptance threshold | 0.3 |
| `VDIFF_GPU_MEMORY_UTILIZATION` | GPU memory fraction | 0.9 |

## Monitoring

### Prometheus Metrics

Metrics are scraped automatically by OpenShift monitoring:

```bash
# Check metrics endpoint
curl https://${ROUTE}/metrics
```

### Grafana Dashboard

Import the vdiff dashboard from `deploy/grafana/vdiff-dashboard.json` into your Grafana instance.

## Troubleshooting

### Pod Not Starting

```bash
# Check events
oc describe pod -l serving.kserve.io/inferenceservice=llada-8b-instruct

# Check GPU availability
oc describe node | grep -A5 "Allocated resources"
```

### Out of Memory

Reduce `VDIFF_GPU_MEMORY_UTILIZATION` or use smaller model:
```yaml
env:
  - name: VDIFF_GPU_MEMORY_UTILIZATION
    value: "0.8"
```

### Slow Cold Start

Use PVC with pre-downloaded model instead of HuggingFace URI.

#!/bin/bash
# Deploy TinyLlama with dfastllm to OCP without pre-built image
# Uses pip wheel served via ConfigMap or direct source mount
#
# Usage:
#   ./scripts/deploy-tinyllama-ocp.sh
#   ./scripts/deploy-tinyllama-ocp.sh --cpu-only
#   ./scripts/deploy-tinyllama-ocp.sh --namespace my-project

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
NAMESPACE="${NAMESPACE:-default}"
CPU_ONLY=false
MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Parse args
while [[ $# -gt 0 ]]; do
  case $1 in
    --cpu-only) CPU_ONLY=true; shift ;;
    --namespace) NAMESPACE="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

echo "=== Building dfastllm wheel ==="
cd "$PROJECT_ROOT"
pip wheel --no-deps -w /tmp/dfastllm-wheel .
WHEEL_FILE=$(ls /tmp/dfastllm-wheel/dfastllm-*.whl | head -1)
echo "Built: $WHEEL_FILE"

echo "=== Creating ConfigMap with dfastllm source ==="
oc create configmap dfastllm-source \
  --from-file=dfastllm/="$PROJECT_ROOT/dfastllm/" \
  --from-file=pyproject.toml="$PROJECT_ROOT/pyproject.toml" \
  --from-file=setup.py="$PROJECT_ROOT/setup.py" \
  --from-file=requirements.txt="$PROJECT_ROOT/requirements.txt" \
  -n "$NAMESPACE" \
  --dry-run=client -o yaml | oc apply -f -

echo "=== Deploying TinyLlama pod ==="
if [ "$CPU_ONLY" = true ]; then
  PYTORCH_IMAGE="pytorch/pytorch:2.4.0-cpu"
  GPU_RESOURCE=""
else
  PYTORCH_IMAGE="pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime"
  GPU_RESOURCE="nvidia.com/gpu: \"1\""
fi

cat <<EOF | oc apply -n "$NAMESPACE" -f -
apiVersion: v1
kind: Pod
metadata:
  name: tinyllama-dfastllm
  labels:
    app: tinyllama-dfastllm
spec:
  containers:
    - name: dfastllm
      image: $PYTORCH_IMAGE
      command: ["/bin/bash", "-c"]
      args:
        - |
          cd /app
          pip install --no-cache-dir -r requirements.txt
          pip install --no-cache-dir -e .
          python -m dfastllm.entrypoints.openai.api_server \\
            --model $MODEL \\
            --host 0.0.0.0 \\
            --port 8000 \\
            --trust-remote-code
      ports:
        - containerPort: 8000
      env:
        - name: HF_HOME
          value: /cache/huggingface
        - name: PYTHONUNBUFFERED
          value: "1"
      resources:
        requests:
          cpu: "2"
          memory: "4Gi"
          $GPU_RESOURCE
        limits:
          cpu: "4"
          memory: "8Gi"
          $GPU_RESOURCE
      volumeMounts:
        - name: source
          mountPath: /app
        - name: cache
          mountPath: /cache
        - name: shm
          mountPath: /dev/shm
  volumes:
    - name: source
      configMap:
        name: dfastllm-source
    - name: cache
      emptyDir:
        sizeLimit: 10Gi
    - name: shm
      emptyDir:
        medium: Memory
        sizeLimit: 2Gi
  restartPolicy: Always
---
apiVersion: v1
kind: Service
metadata:
  name: tinyllama-dfastllm
spec:
  selector:
    app: tinyllama-dfastllm
  ports:
    - port: 8000
      targetPort: 8000
EOF

echo "=== Deployment complete ==="
echo ""
echo "Wait for pod to be ready:"
echo "  oc get pods -l app=tinyllama-dfastllm -w"
echo ""
echo "Check logs:"
echo "  oc logs -f tinyllama-dfastllm"
echo ""
echo "Port forward to test:"
echo "  oc port-forward svc/tinyllama-dfastllm 8000:8000"
echo "  curl http://localhost:8000/v1/models"


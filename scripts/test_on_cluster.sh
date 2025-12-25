#!/bin/bash
# =============================================================================
# dfastllm Cluster Test Script
# =============================================================================
# This script builds, deploys, and tests dfastllm on OpenShift
#
# Usage: ./scripts/test_on_cluster.sh
# =============================================================================

set -e

echo "============================================================"
echo "           dfastllm Cluster Test Script                        "
echo "============================================================"

# Check if logged in
if ! oc whoami &>/dev/null; then
    echo "❌ Not logged in to OpenShift. Please run:"
    echo "   oc login --token=<token> --server=<server>"
    exit 1
fi

echo "✅ Logged in as: $(oc whoami)"
echo ""

# Variables
PROJECT="dfastllm-test"
IMAGE="quay.io/mwaykole/dfastllm:test-$(date +%Y%m%d-%H%M%S)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Step 1: Create project if needed
echo "📦 Setting up project..."
oc get project "$PROJECT" &>/dev/null || oc new-project "$PROJECT"
oc project "$PROJECT"

# Step 2: Build Docker image
echo ""
echo "🔨 Building Docker image..."
podman build -t "$IMAGE" -f Dockerfile.vllm .

# Step 3: Push to quay.io
echo ""
echo "📤 Pushing image to quay.io..."
podman push "$IMAGE"

echo ""
echo "✅ Image pushed: $IMAGE"

# Step 4: Update and deploy ServingRuntime
echo ""
echo "🚀 Deploying ServingRuntime..."
cat <<EOF | oc apply -f -
apiVersion: serving.kserve.io/v1alpha1
kind: ServingRuntime
metadata:
  name: dfastllm-runtime-test
spec:
  containers:
    - name: kserve-container
      image: $IMAGE
      args:
        - "--model"
        - "{{.Model}}"
        - "--port"
        - "8000"
        - "--trust-remote-code"
        - "--flash-attention"
        - "--enable-apd"
      resources:
        requests:
          nvidia.com/gpu: "1"
          memory: "16Gi"
        limits:
          nvidia.com/gpu: "1"
          memory: "32Gi"
      env:
        - name: HF_HOME
          value: /tmp/.cache
        - name: TRANSFORMERS_CACHE
          value: /tmp/.cache
  supportedModelFormats:
    - name: diffusion-llm
      autoSelect: true
    - name: dfastllm
      autoSelect: true
  protocolVersions:
    - v1
    - v2
EOF

# Step 5: Deploy test InferenceService with Phi-2 (smaller model)
echo ""
echo "🚀 Deploying InferenceService (Phi-2 model)..."
cat <<EOF | oc apply -f -
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: dfastllm-test
  annotations:
    serving.kserve.io/deploymentMode: RawDeployment
spec:
  predictor:
    model:
      modelFormat:
        name: dfastllm
      runtime: dfastllm-runtime-test
      storageUri: pvc://model-pvc-large/models/phi-2
      resources:
        requests:
          nvidia.com/gpu: "1"
          memory: "16Gi"
        limits:
          nvidia.com/gpu: "1"
          memory: "16Gi"
      args:
        - "--model"
        - "/models/phi-2"
        - "--flash-attention"
        - "--trust-remote-code"
      env:
        - name: HF_HOME
          value: "/tmp/.cache"
        - name: TRANSFORMERS_CACHE
          value: "/tmp/.cache"
EOF

# Step 6: Wait for deployment
echo ""
echo "⏳ Waiting for deployment to be ready..."
for i in {1..60}; do
    STATUS=$(oc get inferenceservice dfastllm-test -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' 2>/dev/null || echo "Unknown")
    if [[ "$STATUS" == "True" ]]; then
        echo "✅ Deployment ready!"
        break
    fi
    echo "  Waiting... ($i/60)"
    sleep 10
done

# Step 7: Get route
echo ""
echo "🔗 Getting route..."
ROUTE_HOST=$(oc get route dfastllm-test-predictor -o jsonpath='{.spec.host}' 2>/dev/null || echo "")

if [[ -z "$ROUTE_HOST" ]]; then
    echo "Creating route..."
    oc expose service dfastllm-test-predictor --port=80 2>/dev/null || true
    sleep 5
    ROUTE_HOST=$(oc get route dfastllm-test-predictor -o jsonpath='{.spec.host}' 2>/dev/null || echo "")
fi

if [[ -z "$ROUTE_HOST" ]]; then
    echo "❌ Could not get route. Trying via service..."
    # Use port-forward as fallback
    oc port-forward svc/dfastllm-test-predictor 8000:80 &
    PF_PID=$!
    sleep 5
    VDIFF_URL="http://localhost:8000"
else
    VDIFF_URL="http://${ROUTE_HOST}"
fi

echo "📍 dfastllm URL: $VDIFF_URL"

# Step 8: Run tests
echo ""
echo "🧪 Running tests..."
echo ""

# Test 1: Health check
echo "Test 1: Health Check"
HEALTH=$(curl -s -o /dev/null -w "%{http_code}" "$VDIFF_URL/health" 2>/dev/null || echo "000")
if [[ "$HEALTH" == "200" ]]; then
    echo "  ✅ Health check passed"
else
    echo "  ❌ Health check failed (HTTP $HEALTH)"
fi

# Test 2: Models endpoint
echo ""
echo "Test 2: Models Endpoint"
MODELS=$(curl -s "$VDIFF_URL/v1/models" 2>/dev/null || echo "{}")
echo "  Response: $MODELS" | head -c 200
echo ""

# Test 3: Completion
echo ""
echo "Test 3: Text Completion"
COMPLETION=$(curl -s -X POST "$VDIFF_URL/v1/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "phi-2",
        "prompt": "Hello, how are you?",
        "max_tokens": 20,
        "temperature": 0.7
    }' 2>/dev/null || echo "{}")
echo "  Response: $COMPLETION" | head -c 300
echo ""

# Test 4: Chat Completion
echo ""
echo "Test 4: Chat Completion"
CHAT=$(curl -s -X POST "$VDIFF_URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "phi-2",
        "messages": [{"role": "user", "content": "Say hello!"}],
        "max_tokens": 20
    }' 2>/dev/null || echo "{}")
echo "  Response: $CHAT" | head -c 300
echo ""

# Test 5: Streaming
echo ""
echo "Test 5: Streaming Completion"
echo "  Testing stream..."
curl -s -X POST "$VDIFF_URL/v1/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "phi-2",
        "prompt": "Count to 5:",
        "max_tokens": 30,
        "stream": true
    }' 2>/dev/null | head -c 500
echo ""

# Test 6: Error handling
echo ""
echo "Test 6: Error Handling (invalid model)"
ERROR=$(curl -s -X POST "$VDIFF_URL/v1/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "nonexistent-model",
        "prompt": "test",
        "max_tokens": 10
    }' 2>/dev/null || echo "{}")
if echo "$ERROR" | grep -q "error"; then
    echo "  ✅ Error handling works"
    echo "  Response: $ERROR" | head -c 200
else
    echo "  ❌ Error handling issue"
fi
echo ""

# Test 7: Metrics
echo ""
echo "Test 7: Prometheus Metrics"
METRICS=$(curl -s "$VDIFF_URL/metrics" 2>/dev/null | head -20 || echo "Failed")
if echo "$METRICS" | grep -q "dfastllm_\|http_"; then
    echo "  ✅ Metrics available"
else
    echo "  ⚠️ Metrics may not be available"
fi

# Test 8: Throughput benchmark
echo ""
echo "Test 8: Throughput Benchmark"
echo "  Running 3 requests with different token lengths..."
for TOKENS in 16 32 64; do
    START=$(date +%s.%N)
    RESP=$(curl -s -X POST "$VDIFF_URL/v1/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"phi-2\",
            \"prompt\": \"Explain the concept of\",
            \"max_tokens\": $TOKENS
        }" 2>/dev/null || echo "{}")
    END=$(date +%s.%N)
    LATENCY=$(echo "$END - $START" | bc)
    THROUGHPUT=$(echo "scale=1; $TOKENS / $LATENCY" | bc)
    echo "    $TOKENS tokens: ${LATENCY}s latency, ${THROUGHPUT} tok/s"
done

# Cleanup port-forward if used
if [[ -n "$PF_PID" ]]; then
    kill $PF_PID 2>/dev/null || true
fi

# Summary
echo ""
echo "============================================================"
echo "                    TEST SUMMARY                            "
echo "============================================================"
echo "Image: $IMAGE"
echo "Project: $PROJECT"
echo "URL: $VDIFF_URL"
echo ""
echo "All tests completed! Check results above."
echo "============================================================"


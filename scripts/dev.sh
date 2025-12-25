#!/bin/bash
# dfastllm Development Helper Script
# Usage: ./scripts/dev.sh [command]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DEV_POD="dfastllm-dev"

cd "$PROJECT_DIR"

case "${1:-help}" in
  local)
    echo "🖥️  Starting local development server..."
    echo "   Model will be downloaded if not cached"
    pip install -e . 2>/dev/null || true
    python -m dfastllm.entrypoints.openai.api_server \
      --model "${2:-microsoft/phi-2}" \
      --port 8000 \
      --trust-remote-code
    ;;

  test)
    echo "🧪 Running unit tests..."
    pip install -e ".[dev]" 2>/dev/null || true
    python -m pytest tests/unit/ -v "${@:2}"
    ;;

  dev-pod)
    echo "🚀 Creating/connecting to dev pod on cluster..."
    if oc get pod $DEV_POD &>/dev/null; then
      echo "   Connecting to existing dev pod..."
    else
      echo "   Creating new dev pod..."
      oc apply -f deploy/kubernetes/dev-pod.yaml
      echo "   Waiting for pod to be ready..."
      oc wait --for=condition=Ready pod/$DEV_POD --timeout=300s
    fi
    oc exec -it $DEV_POD -- bash
    ;;

  sync)
    echo "📤 Syncing code to dev pod..."
    if ! oc get pod $DEV_POD &>/dev/null; then
      echo "❌ Dev pod not found. Run: ./scripts/dev.sh dev-pod"
      exit 1
    fi
    # Sync entire project (dfastllm source + setup files)
    echo "   Cleaning old files..."
    oc exec $DEV_POD -- rm -rf /tmp/dfastllm-src 2>/dev/null || true
    oc exec $DEV_POD -- mkdir -p /tmp/dfastllm-src
    echo "   Copying source code..."
    oc cp dfastllm $DEV_POD:/tmp/dfastllm-src/dfastllm
    oc cp pyproject.toml $DEV_POD:/tmp/dfastllm-src/pyproject.toml
    oc cp setup.py $DEV_POD:/tmp/dfastllm-src/setup.py 2>/dev/null || true
    echo "   Installing package..."
    oc exec $DEV_POD -- sh -c "HOME=/tmp pip install --user -e /tmp/dfastllm-src --quiet" 2>/dev/null || \
      oc exec $DEV_POD -- sh -c "HOME=/tmp pip install --user -e /tmp/dfastllm-src"
    echo "✅ Code synced! Run server with:"
    echo "   ./scripts/dev.sh run"
    ;;

  run)
    echo "▶️  Running server in dev pod..."
    if ! oc get pod $DEV_POD &>/dev/null; then
      echo "❌ Dev pod not found. Run: ./scripts/dev.sh dev-pod"
      exit 1
    fi
    oc exec -it $DEV_POD -- python -m dfastllm.entrypoints.openai.api_server \
      --model "${2:-/models/phi-2}" \
      --port 8000 \
      --trust-remote-code
    ;;

  forward)
    echo "🔗 Port-forwarding dev pod to localhost:8000..."
    oc port-forward pod/$DEV_POD 8000:8000
    ;;

  cleanup)
    echo "🧹 Cleaning up dev pod..."
    oc delete pod $DEV_POD --ignore-not-found
    echo "✅ Done"
    ;;

  build)
    echo "🔨 Building Docker image..."
    podman build -t quay.io/mwaykole/dfastllm:dev -f Dockerfile.vllm .
    echo "✅ Image built: quay.io/mwaykole/dfastllm:dev"
    ;;

  push)
    echo "📤 Pushing Docker image..."
    podman push quay.io/mwaykole/dfastllm:dev
    echo "✅ Image pushed"
    ;;

  help|*)
    cat << EOF
dfastllm Development Helper

Usage: ./scripts/dev.sh <command> [options]

Commands:
  local [model]    Run server locally (default: microsoft/phi-2)
  test [args]      Run unit tests
  dev-pod          Create/connect to development pod on cluster
  sync             Sync local code to dev pod
  run [model]      Run server in dev pod (default: /models/phi-2)
  forward          Port-forward dev pod to localhost:8000
  cleanup          Delete dev pod
  build            Build Docker image locally
  push             Push Docker image to registry
  help             Show this help

Development Workflow (Fast):
  1. ./scripts/dev.sh dev-pod     # Create dev pod (one-time)
  2. # Make code changes locally
  3. ./scripts/dev.sh sync        # Sync changes to pod
  4. ./scripts/dev.sh run         # Test on cluster GPU
  5. ./scripts/dev.sh forward     # (In another terminal) Access locally

Local Development (Fastest, no GPU):
  1. ./scripts/dev.sh local       # Run locally
  2. ./scripts/dev.sh test        # Run unit tests

EOF
    ;;
esac


#!/bin/bash
# Build script for vdiff Serving

set -e

echo "Building vdiff Serving..."

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf build/ dist/ *.egg-info/

# Build package
echo "Building Python package..."
python -m build

# Build Docker image (optional)
if [ "$1" == "--docker" ]; then
    echo "Building Docker image..."
    docker build -t vdiff-serving:latest -f deploy/docker/Dockerfile .
    echo "Docker image built: vdiff-serving:latest"
fi

echo "Build complete!"
echo ""
echo "Artifacts:"
ls -la dist/

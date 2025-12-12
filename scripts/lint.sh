#!/bin/bash
# Lint script for vdiff Serving

set -e

echo "Running linters for vdiff Serving..."

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Parse arguments
FIX=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --fix)
            FIX=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Running ruff..."
if [ "$FIX" = true ]; then
    ruff check --fix vdiff tests
else
    ruff check vdiff tests
fi

echo ""
echo "Running black..."
if [ "$FIX" = true ]; then
    black vdiff tests
else
    black --check vdiff tests
fi

echo ""
echo "Running isort..."
if [ "$FIX" = true ]; then
    isort vdiff tests
else
    isort --check-only vdiff tests
fi

echo ""
echo "Running mypy..."
mypy vdiff --ignore-missing-imports

echo ""
echo "All linting checks passed!"

# vdiff Makefile
# Production-ready development automation

.PHONY: help install dev test lint format check clean build docker serve

# Default target
help:
	@echo "vdiff - vLLM-Compatible Serving for Diffusion LLMs"
	@echo "==================================================="
	@echo ""
	@echo "Setup:"
	@echo "  make install      Install package"
	@echo "  make dev          Install with dev dependencies"
	@echo ""
	@echo "Development:"
	@echo "  make test         Run tests"
	@echo "  make lint         Run linter"
	@echo "  make format       Format code"
	@echo "  make check        Run all checks (lint + test)"
	@echo ""
	@echo "Build:"
	@echo "  make build        Build Python package"
	@echo "  make docker       Build Docker image (GPU)"
	@echo "  make docker-cpu   Build Docker image (CPU-only)"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-up    Start with docker-compose (GPU)"
	@echo "  make docker-down  Stop docker-compose"
	@echo "  make docker-logs  View container logs"
	@echo ""
	@echo "Run:"
	@echo "  make serve        Start local server"
	@echo "  make serve-llada  Start with LLaDA model"
	@echo ""
	@echo "Clean:"
	@echo "  make clean        Remove build artifacts"

# ============================================================================
# Installation
# ============================================================================

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

# ============================================================================
# Testing
# ============================================================================

test:
	pytest tests/ -v --tb=short

test-unit:
	pytest tests/unit/ -v --tb=short

test-integration:
	pytest tests/integration/ -v --tb=short -m integration

test-coverage:
	pytest tests/ -v --cov=vdiff --cov-report=html --cov-report=term-missing

# ============================================================================
# Code Quality
# ============================================================================

lint:
	ruff check vdiff/ tests/
	mypy vdiff/ --ignore-missing-imports

format:
	black vdiff/ tests/
	ruff check --fix vdiff/ tests/

check: lint test
	@echo "All checks passed!"

# ============================================================================
# Build
# ============================================================================

build:
	pip install build
	python -m build

# Docker builds (matching vLLM style)
docker:
	docker build -t vdiff:latest .

docker-cpu:
	docker build --build-arg USE_CUDA=0 -t vdiff:cpu .

docker-cuda:
	docker build --build-arg USE_CUDA=1 --build-arg CUDA_VERSION=12.4.1 -t vdiff:cuda .

# Docker Compose
docker-up:
	docker-compose up -d

docker-up-cpu:
	docker-compose --profile cpu up -d vdiff-cpu

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f vdiff

# ============================================================================
# Run
# ============================================================================

# Default model for local testing
MODEL ?= gpt2

serve:
	python -m vdiff.entrypoints.openai.api_server \
		--model $(MODEL) \
		--port 8000

serve-dev:
	uvicorn vdiff.entrypoints.openai.api_server:create_app \
		--reload \
		--host 0.0.0.0 \
		--port 8000

serve-llada:
	python -m vdiff.entrypoints.openai.api_server \
		--model GSAI-ML/LLaDA-8B-Instruct \
		--port 8000 \
		--trust-remote-code

# ============================================================================
# Clean
# ============================================================================

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# ============================================================================
# Documentation
# ============================================================================

docs:
	pip install -e ".[docs]"
	mkdocs serve

docs-build:
	mkdocs build

# ============================================================================
# Release
# ============================================================================

release-check:
	@echo "Checking release readiness..."
	@make check
	@echo "Version: $(shell python -c 'from vdiff.version import __version__; print(__version__)')"
	@echo "Ready for release!"

release-build: clean
	pip install build twine
	python -m build
	twine check dist/*

# ============================================================================
# Docker Compose (for development)
# ============================================================================

compose-up:
	docker-compose up -d

compose-down:
	docker-compose down

compose-logs:
	docker-compose logs -f vdiff

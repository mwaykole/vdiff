# vdiff Serving Makefile
# Common development tasks

.PHONY: help install dev lint format test test-cov build run clean docker-build docker-run

# Default target
help:
	@echo "vdiff Serving - Development Commands"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  install     Install production dependencies"
	@echo "  dev         Install development dependencies"
	@echo "  lint        Run linters (ruff, mypy)"
	@echo "  format      Format code (black, isort)"
	@echo "  test        Run tests"
	@echo "  test-cov    Run tests with coverage"
	@echo "  build       Build package"
	@echo "  run         Run the server locally"
	@echo "  clean       Clean build artifacts"
	@echo "  docker-build Build Docker image"
	@echo "  docker-run  Run Docker container"

# Installation
install:
	pip install -r requirements.txt
	pip install -e .

dev:
	pip install -r requirements-dev.txt
	pip install -e .
	pre-commit install

# Linting and formatting
lint:
	ruff check vdiff tests
	mypy vdiff --ignore-missing-imports

format:
	black vdiff tests
	ruff check --fix vdiff tests
	isort vdiff tests

# Testing
test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=vdiff --cov-report=term-missing --cov-report=html

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

test-compat:
	pytest tests/compatibility/ -v

# Building
build:
	python -m build

build-wheel:
	pip wheel --no-deps -w dist .

# Running
run:
	python -m vdiff.entrypoints.openai.api_server --model $(MODEL) --port 8000

run-dev:
	uvicorn vdiff.entrypoints.openai.api_server:create_app --host 0.0.0.0 --port 8000 --reload --factory

# Docker
docker-build:
	docker build -t vdiff-serving:latest -f deploy/docker/Dockerfile .

docker-build-dev:
	docker build -t vdiff-serving:dev -f deploy/docker/Dockerfile.dev .

docker-run:
	docker run --gpus all -p 8000:8000 -e MODEL_NAME=$(MODEL) vdiff-serving:latest

# Cleanup
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Documentation
docs:
	mkdocs serve

docs-build:
	mkdocs build

# Release helpers
version:
	@python -c "from vdiff.version import __version__; print(__version__)"

tag:
	git tag v$(shell make version)
	git push origin v$(shell make version)

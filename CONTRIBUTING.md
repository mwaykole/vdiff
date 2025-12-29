# Contributing to dfastllm Serving

Thank you for your interest in contributing to dfastllm Serving! This document provides guidelines and information for contributors.

## Getting Started

### Development Setup

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/dfastllm-serving.git
   cd dfastllm-serving
   ```
3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
4. Install development dependencies:
   ```bash
   make dev
   ```
5. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

### Running Tests

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run specific test types
make test-unit
make test-integration
make test-compat
```

### Code Style

We use the following tools for code quality:

- **Black** for code formatting
- **isort** for import sorting
- **ruff** for linting
- **mypy** for type checking

Run all checks with:
```bash
make lint
```

Auto-fix issues with:
```bash
make format
```

## Making Changes

### Branch Naming

- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation updates
- `refactor/description` - Code refactoring

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat: add new feature`
- `fix: resolve bug`
- `docs: update documentation`
- `test: add tests`
- `refactor: restructure code`
- `chore: maintenance tasks`

### Pull Request Process

1. Create a feature branch from `main`
2. Make your changes
3. Add/update tests as needed
4. Ensure all tests pass
5. Update documentation if needed
6. Submit a pull request

### Pull Request Checklist

- [ ] Tests pass (`make test`)
- [ ] Linting passes (`make lint`)
- [ ] Documentation updated (if applicable)
- [ ] CHANGELOG.md updated (for user-facing changes)
- [ ] Commit messages follow conventions

## vLLM Compatibility

dfastllm aims to maintain 100% API compatibility with vLLM. When making changes:

1. Ensure request/response formats match vLLM exactly
2. Maintain CLI argument compatibility
3. Keep endpoint paths identical
4. Match Prometheus metric naming conventions

Run compatibility tests:
```bash
make test-compat
```

## Adding New Features

### dfastllm-Specific Extensions

When adding dfastllm-specific features (not in vLLM):

1. Make them **optional** - don't break vLLM compatibility
2. Use separate request/response fields
3. Document the extension clearly
4. Add appropriate tests

### Fast-dfastllm Optimizations

For performance optimizations:

1. Ensure they're disabled by default or backward compatible
2. Add configuration options
3. Include benchmarks showing improvement
4. Document the tradeoffs

## Reporting Issues

### Bug Reports

Include:
- dfastllm version
- Python version
- GPU/CUDA version
- Minimal reproduction code
- Expected vs actual behavior
- Full error traceback

### Feature Requests

Include:
- Use case description
- Proposed API/behavior
- Compatibility considerations

## Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.

## Questions?

- Open a GitHub issue for bugs/features
- Start a discussion for questions
- Check existing issues before creating new ones

Thank you for contributing! 🎉

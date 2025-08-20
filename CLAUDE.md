# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Ragas is an evaluation toolkit for Large Language Model (LLM) applications. It provides objective metrics for evaluating LLM applications, test data generation capabilities, and integrations with popular LLM frameworks.

The repository contains:

1. **Ragas Library** - The main evaluation toolkit including experimental features (in `/ragas` directory)
   - Core evaluation metrics and test generation
   - Experimental features available at `ragas.experimental`

## Development Environment Setup

### Installation Options

Choose the appropriate installation based on your development needs:

#### Minimal Development Setup
Recommended for:
- Code formatting, linting, and type checking
- Running unit tests
- Basic development tasks
- Contributing to core functionality

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install minimal development dependencies
pip install -U setuptools  # Required on newer Python versions
pip install -e "./ragas[dev]"
# Or use make command
make install-minimal
```

#### Full Development Setup
Required for:
- Working with experimental features
- ML model development and testing
- Notebook development
- Tracing and observability features
- Complete testing including benchmarks

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install full development dependencies
pip install -U setuptools
uv sync --group dev-full --package ragas
# Or use make command
make install
```

## Common Commands

### Commands (from root directory)

```bash
# Setup and installation
make install         # Install full development dependencies (including ML stack)
make install-minimal # Install minimal development dependencies

# Code quality
make format         # Format and lint all code
make type           # Type check all code
make check          # Quick health check (format + type, no tests)

# Testing
make test           # Run all unit tests (including experimental)
make test-e2e       # Run end-to-end tests

# CI/Build
make run-ci         # Run complete CI pipeline
make clean          # Clean all generated files

# Documentation
make build-docs     # Build all documentation
make serve-docs     # Serve documentation locally

# Benchmarks
make benchmarks     # Run performance benchmarks
make benchmarks-docker # Run benchmarks in Docker
```

### Project-Specific Commands

The ragas directory has its own Makefile for focused development:

```bash
# Ragas development (from ragas/ directory)
cd ragas
make format         # Format ragas code only
make type           # Type check ragas code only
make check          # Quick format + type check
make test           # Run all tests (core + experimental)
make run-ci         # Run ragas CI pipeline
```

### Testing

```bash
# Run all tests (from root)
make test

# Run specific test (using pytest -k flag)
make test k="test_name"

# Run end-to-end tests
make test-e2e

# Run tests from ragas directory
cd ragas && make test           # Run all ragas tests (core + experimental)

# Direct pytest commands for more control
cd ragas && uv run pytest tests/unit -k "test_name"
cd ragas && uv run pytest tests/experimental -v
```

### Documentation

```bash
# Build all documentation (from root)
make build-docs

# Serve documentation locally
make serve-docs

# Process experimental notebooks
make process-experimental-notebooks
```

### Benchmarks

```bash
# Run all benchmarks locally
make benchmarks

# Run benchmarks in Docker
make benchmarks-docker
```

## Project Architecture

The repository has the following structure:

```
/
├── ragas/           # Main ragas project
│   ├── src/ragas/   # Source code including experimental features
│   │   ├── experimental/  # Experimental features
│   ├── tests/       # All tests (core + experimental)
│   │   ├── experimental/  # Experimental tests
│   ├── examples/    # Example code
│   │   ├── experimental/  # Experimental examples
│   ├── pyproject.toml  # Unified build config
│
├── docs/            # Documentation
├── scripts/         # Build/CI scripts
├── workspace.toml   # Root project config (for dev tools)
├── Makefile         # Build commands
└── README.md        # Repository overview
```

### Ragas Core Components

The Ragas core library provides metrics, test data generation and evaluation functionality for LLM applications:

1. **Metrics** - Various metrics for evaluating LLM applications including:

   - AspectCritic
   - AnswerCorrectness
   - ContextPrecision
   - ContextRecall
   - Faithfulness
   - and many more

2. **Test Data Generation** - Automatic creation of test datasets for LLM applications

3. **Integrations** - Integrations with popular LLM frameworks like LangChain, LlamaIndex, and observability tools

### Experimental Components

The experimental features are now integrated into the main ragas package:

1. **Experimental features** are available at `ragas.experimental`
2. **Dataset and Experiment management** - Enhanced data handling for experiments
3. **Advanced metrics** - Extended metric capabilities
4. **Backend support** - Multiple storage backends (CSV, JSONL, Google Drive, in-memory)

To use experimental features:

```python
from ragas.experimental import Dataset
from ragas import experiment
from ragas.backends import get_registry
```

## Debugging Logs

To view debug logs for any module:

```python
import logging

# Configure logging for a specific module (example with analytics)
analytics_logger = logging.getLogger('ragas._analytics')
analytics_logger.setLevel(logging.DEBUG)

# Create a console handler and set its level
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# Create a formatter and add it to the handler
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# Add the handler to the logger
analytics_logger.addHandler(console_handler)
```

## Memories

- whenever you create such docs put in in /\_experiments because that is gitignored and you can use it as a scratchpad or tmp directory for storing these
- always use uv to run python and python related commandline tools like isort, ruff, pyright ect. This is because we are using uv to manage the .venv and dependencies.

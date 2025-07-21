# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Ragas is an evaluation toolkit for Large Language Model (LLM) applications. It provides objective metrics for evaluating LLM applications, test data generation capabilities, and integrations with popular LLM frameworks.

The repository is structured as a monorepo containing:
1. **Ragas Core Library** - The main evaluation toolkit (in `/ragas` directory)
2. **Ragas Experimental** - An nbdev-based project for Ragas extensions (in `/experimental` directory)

## Development Environment Setup

### Installation

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# For ragas core
pip install -U setuptools  # Required on newer Python versions
pip install -e ".[dev]"

# For experimental project
pip install -e "./experimental[dev]"
```

## Common Commands

### Monorepo-Wide Commands (from root directory)

```bash
# Setup and installation
make install        # Install dependencies for both projects

# Code quality (runs on both ragas/ and experimental/)
make format         # Format and lint all code
make type           # Type check all code
make check          # Quick health check (format + type, no tests)

# Testing
make test           # Run all unit tests
make test-e2e       # Run end-to-end tests

# CI/Build
make run-ci         # Run complete CI pipeline for both projects
make clean          # Clean all generated files

# Documentation
make build-docs     # Build all documentation
make serve-docs     # Serve documentation locally

# Benchmarks
make benchmarks     # Run performance benchmarks
make benchmarks-docker # Run benchmarks in Docker
```

### Project-Specific Commands

Each project directory (`ragas/` and `experimental/`) has its own Makefile with core development commands:

```bash
# Ragas core development (from ragas/ directory)
cd ragas
make format         # Format ragas code only
make type           # Type check ragas code only
make check          # Quick format + type check
make test           # Run ragas tests only
make run-ci         # Run ragas CI pipeline only

# Experimental development (from experimental/ directory)
cd experimental
make format         # Format experimental code only
make type           # Type check experimental code only
make check          # Quick format + type check
make test           # Run experimental tests only
make run-ci         # Run experimental CI pipeline only
```

### Testing

```bash
# Run all tests in the monorepo (from root)
make test

# Run specific test (using pytest -k flag)
make test k="test_name"

# Run end-to-end tests
make test-e2e

# Run tests for specific projects
cd ragas && make test           # Run ragas tests only
cd experimental && make test    # Run experimental tests only

# Direct pytest commands for more control
cd ragas && uv run pytest tests/unit -k "test_name"
cd experimental && uv run pytest -v
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

The monorepo has the following structure:

```
/
├── ragas/           # Main ragas project
│   ├── src/         # Original source code
│   ├── tests/       # Original tests
│   ├── pyproject.toml  # ragas-specific build config
│
├── experimental/    # nbdev-based experimental project
│   ├── nbs/         # Notebooks for nbdev  
│   ├── ragas_experimental/  # Generated code
│   ├── pyproject.toml  # experimental-specific config
│   ├── settings.ini    # nbdev config
│
├── docs/            # Combined documentation
├── scripts/         # Shared build/CI scripts
├── workspace.toml   # Root project config (for dev tools)
├── Makefile         # Combined build commands
└── README.md        # Monorepo overview
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

The experimental package (`ragas_experimental`) is for developing new features and extensions using nbdev:

1. When working on the experimental project, make changes in the notebook files in `experimental/nbs/`
2. Run `nbdev_export` to generate Python code in `experimental/ragas_experimental/`
3. Run tests with `pytest` in the experimental directory
4. Generate docs with `nbdev_docs`

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

- whenever you create such docs put in in /_experiments because that is gitignored and you can use it as a scratchpad or tmp directory for storing these
- always use uv to run python and python related commandline tools like isort, ruff, pyright ect. This is because we are using uv to manage the .venv and dependencies.

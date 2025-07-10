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

### Build and Development

```bash
# Format code (ragas core)
make format

# Format code (experimental)
make format-experimental 

# Format all code in the monorepo
make format-all

# Lint code (ragas core)
make lint

# Lint code (experimental)
make lint-experimental

# Lint all code in the monorepo
make lint-all

# Type check code (ragas core)
make type

# Type check code (experimental)
make type-experimental

# Type check all code in the monorepo
make type-all

# Run all CI checks for ragas core
make run-ci

# Run all CI checks for experimental
make run-ci-experimental

# Run all CI checks for both projects
make run-ci-all
```

### Testing

```bash
# Run ragas core tests
make test

# Run specific test (using pytest -k flag)
make test k="test_name"

# Run ragas end-to-end tests
make test-e2e

# Run experimental tests
make test-experimental

# Run all tests in the monorepo
make test-all
```

### Documentation

```bash
# Build ragas documentation
make build-docsite-ragas

# Build experimental documentation
make build-docsite-experimental

# Build all documentation
make build-docsite

# Serve documentation locally
make serve-docsite
```

### Benchmarks

```bash
# Run benchmarks for Evaluation
make run-benchmarks-eval

# Run benchmarks for TestSet Generation
make run-benchmarks-testset

# Run benchmarks in docker
make run-benchmarks-in-docker
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

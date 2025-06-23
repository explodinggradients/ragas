GIT_ROOT ?= $(shell git rev-parse --show-toplevel)

# Optionally show commands being executed with V=1
Q := $(if $(V),,@)

# Common paths
RAGAS_PATHS := ragas/src ragas/tests docs
EXPERIMENTAL_PATH := experimental/ragas_experimental

help: ## Show all Makefile targets
	$(Q)grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[33m%-30s\033[0m %s\n", $$1, $$2}'

# =============================================================================
# SETUP & INSTALLATION
# =============================================================================

install: ## Install dependencies for both ragas and experimental
	@echo "Installing dependencies..."
	@echo "Installing ragas dependencies..."
	$(Q)uv pip install -e "./ragas[dev]"
	@echo "Installing experimental dependencies..."
	$(Q)uv pip install -e "./experimental[dev]"

setup: install ## Complete development environment setup
	@echo "Development environment setup complete!"
	@echo "Available commands: make help"

# =============================================================================
# CODE QUALITY
# =============================================================================

.PHONY: help install setup format type check clean test test-e2e benchmarks benchmarks-docker run-ci build-docs serve-docs process-experimental-notebooks
format: ## Format and lint all code in the monorepo
	@echo "Formatting and linting all code..."
	@echo "(isort) Ordering imports in ragas..."
	$(Q)cd ragas && uv run isort .
	@echo "(black) Formatting ragas..."
	$(Q)uv run black --config ragas/pyproject.toml $(RAGAS_PATHS)
	@echo "(black) Formatting stubs..."
	$(Q)find ragas/src -name "*.pyi" ! -name "*_pb2*" -exec uv run black --pyi --config ragas/pyproject.toml {} \;
	@echo "(ruff) Auto-fixing ragas (includes unused imports)..."
	$(Q)uv run ruff check $(RAGAS_PATHS) --fix-only
	@echo "(ruff) Final linting check for ragas..."
	$(Q)uv run ruff check $(RAGAS_PATHS)
	@echo "(black) Formatting experimental..."
	$(Q)cd experimental && uv run black $(EXPERIMENTAL_PATH)
	@echo "(ruff) Auto-fixing experimental (includes unused imports)..."
	$(Q)cd experimental && uv run ruff check $(EXPERIMENTAL_PATH) --fix-only
	@echo "(ruff) Final linting check for experimental..."
	$(Q)cd experimental && uv run ruff check $(EXPERIMENTAL_PATH)

type: ## Type check all code in the monorepo
	@echo "Type checking all code..."
	@echo "(pyright) Typechecking ragas..."
	$(Q)cd ragas && PYRIGHT_PYTHON_FORCE_VERSION=latest uv run pyright src
	@echo "(pyright) Typechecking experimental..."
	$(Q)PYRIGHT_PYTHON_FORCE_VERSION=latest uv run pyright $(EXPERIMENTAL_PATH)

check: format type ## Quick health check (format + type, no tests)
	@echo "Code quality check complete!"

# =============================================================================
# TESTING
# =============================================================================
benchmarks: ## Run all benchmarks locally
	@echo "Running all benchmarks..."
	@echo "Running evaluation benchmarks..."
	$(Q)cd $(GIT_ROOT)/ragas/tests/benchmarks && uv run python benchmark_eval.py
	@echo "Running testset generation benchmarks..."
	$(Q)cd $(GIT_ROOT)/ragas/tests/benchmarks && uv run python benchmark_testsetgen.py

benchmarks-docker: ## Run benchmarks in docker
	@echo "Running benchmarks in docker..."
	$(Q)cd $(GIT_ROOT) || exit 1
	docker buildx build --build-arg OPENAI_API_KEY=$(OPENAI_API_KEY) -t ragas-benchmark -f $(GIT_ROOT)/ragas/tests/benchmarks/Dockerfile .
	docker inspect ragas-benchmark:latest | jq ".[0].Size" | numfmt --to=si

# =============================================================================
# CI/BUILD
# =============================================================================

run-ci: format type test ## Run complete CI pipeline
	@echo "All CI checks passed!"

clean: ## Clean all generated files
	@echo "Cleaning all generated files..."
	$(Q)cd $(GIT_ROOT)/docs && $(MAKE) clean
	$(Q)find . -type f -name '*.py[co]' -delete -o -type d -name __pycache__ -delete

# =============================================================================
# DOCUMENTATION
# =============================================================================

test: ## Run all unit tests in the monorepo
	@echo "Running all unit tests..."
	@echo "Running ragas tests..."
	$(Q)cd ragas && uv run pytest --nbmake tests/unit $(shell if [ -n "$(k)" ]; then echo "-k $(k)"; fi)
	@echo "Running experimental tests..."
	$(Q)cd experimental && uv run pytest

test-e2e: ## Run all end-to-end tests
	@echo "Running all end-to-end tests..."
	@echo "Running ragas e2e tests..."
	$(Q)cd ragas && uv run pytest --nbmake tests/e2e -s
	@echo "Checking for experimental e2e tests..."
	$(Q)if [ -d "experimental/tests/e2e" ]; then \
		echo "Running experimental e2e tests..."; \
		cd experimental && uv run pytest tests/e2e -s; \
	else \
		echo "No experimental e2e tests found."; \
	fi

# =============================================================================
# BENCHMARKS
# =============================================================================

process-experimental-notebooks: ## Process experimental notebooks to markdown for MkDocs
	@echo "Processing experimental notebooks..."
	$(Q)python $(GIT_ROOT)/scripts/process_experimental_notebooks.py

build-docs: process-experimental-notebooks ## Build all documentation
	@echo "Building all documentation..."
	@echo "Converting ipynb notebooks to md files..."
	$(Q)python $(GIT_ROOT)/docs/ipynb_to_md.py
	@echo "Building ragas documentation..."
	$(Q)mkdocs build
	@echo "Building experimental documentation..."
	$(Q)cd experimental && nbdev_docs

serve-docs: ## Build and serve documentation locally
	$(Q)mkdocs serve --dirtyreload

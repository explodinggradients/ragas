GIT_ROOT ?= $(shell git rev-parse --show-toplevel)

# Optionally show commands being executed with V=1
Q := $(if $(V),,@)

# Common paths
RAGAS_PATHS := ragas/src ragas/tests docs

help: ## Show all Makefile targets
	$(Q)grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[33m%-30s\033[0m %s\n", $$1, $$2}'

# =============================================================================
# SETUP & INSTALLATION
# =============================================================================

install: ## Install full development dependencies for ragas
	@echo "Installing full development dependencies..."
	@echo "Installing ragas with all development dependencies (including ML stack)..."
	$(Q)uv sync --group dev-full --package ragas
	@echo "Setting up pre-commit hooks..."
	$(Q)pre-commit install
	@echo "Full installation complete!"

install-minimal: ## Install minimal development dependencies for ragas
	@echo "Installing minimal development dependencies..."
	@echo "Installing ragas with minimal dev dependencies..."
	$(Q)uv pip install -e "./ragas[dev]"
	@echo "Setting up pre-commit hooks..."
	$(Q)pre-commit install
	@echo "Minimal installation complete!"

# =============================================================================
# CODE QUALITY
# =============================================================================

.PHONY: help install format type check clean test test-e2e benchmarks benchmarks-docker run-ci run-ci-fast run-ci-format-check run-ci-type run-ci-tests build-docs serve-docs process-experimental-notebooks
format: ## Format and lint all code
	@echo "Formatting and linting all code..."
	$(Q)$(MAKE) -C ragas format

type: ## Type check all code
	@echo "Type checking all code..."
	$(Q)$(MAKE) -C ragas type

check: format type ## Quick health check (format + type, no tests)
	@echo "Code quality check complete!"

# =============================================================================
# BENCHMARKS
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

benchmarks-test: ## Run benchmarks for ragas unit tests
	@echo "Running ragas unit tests with timing benchmarks..."
	$(Q)cd ragas && uv run pytest --nbmake tests/unit tests/experimental --durations=0 -v $(shell if [ -n "$(k)" ]; then echo "-k $(k)"; fi)

# =============================================================================
# CI/BUILD
# =============================================================================

run-ci: ## Run complete CI pipeline (mirrors GitHub CI exactly)
	@echo "Running complete CI pipeline..."
	$(Q)$(MAKE) -C ragas run-ci
	@echo "All CI checks passed!"

run-ci-format-check: ## Run format check in dry-run mode (like GitHub CI)
	@echo "Running format check (dry-run, like GitHub CI)..."
	@echo "Checking ragas formatting..."
	$(Q)uv run ruff format --check ragas/src ragas/tests docs
	$(Q)ruff check ragas/src docs ragas/tests

run-ci-type: ## Run type checking (matches GitHub CI)
	@echo "Running type checking (matches GitHub CI)..."
	$(Q)$(MAKE) type

run-ci-tests: ## Run all tests with CI options
	@echo "Running all tests with CI options..."
	$(Q)cd ragas && __RAGAS_DEBUG_TRACKING=true RAGAS_DO_NOT_TRACK=true pytest --nbmake tests/unit tests/experimental --dist loadfile -n auto

run-ci-fast: ## Fast CI check for quick local validation (2-3 minutes)
	@echo "Running fast CI check for quick feedback..."
	@echo "Format check..."
	$(Q)uv run ruff format --check ragas/src ragas/tests docs
	$(Q)ruff check ragas/src docs ragas/tests
	@echo "Core unit tests (no nbmake for speed)..."
	$(Q)cd ragas && pytest tests/unit tests/experimental --dist loadfile -n auto -x
	@echo "Fast CI check completed!"

clean: ## Clean all generated files
	@echo "Cleaning all generated files..."
	$(Q)cd $(GIT_ROOT)/docs && $(MAKE) clean
	$(Q)find . -type f -name '*.py[co]' -delete -o -type d -name __pycache__ -delete

# =============================================================================
# TESTING
# =============================================================================

test: ## Run all unit tests
	@echo "Running all unit tests..."
	$(Q)$(MAKE) -C ragas test $(shell if [ -n "$(k)" ]; then echo "k=$(k)"; fi)

test-all: ## Run all unit tests (including notebooks)
	@echo "Running all unit tests (including notebooks)..."
	$(Q)$(MAKE) -C ragas test-all $(shell if [ -n "$(k)" ]; then echo "k=$(k)"; fi)

test-e2e: ## Run all end-to-end tests
	@echo "Running all end-to-end tests..."
	$(Q)cd ragas && uv run pytest --nbmake tests/e2e -s

# =============================================================================
# DOCUMENTATION
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

serve-docs: ## Build and serve documentation locally
	$(Q)mkdocs serve --dirtyreload

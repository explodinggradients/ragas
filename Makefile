GIT_ROOT ?= $(shell git rev-parse --show-toplevel)

# Optionally show commands being executed with V=1
Q := $(if $(V),,@)

# Common paths - only used for monorepo-wide operations
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

# =============================================================================
# CODE QUALITY
# =============================================================================

.PHONY: help install setup format type check clean test test-e2e benchmarks benchmarks-docker run-ci run-ci-fast run-ci-format-check run-ci-type run-ci-tests build-docs serve-docs process-experimental-notebooks
format: ## Format and lint all code in the monorepo
	@echo "Formatting and linting all code..."
	@echo "Formatting ragas..."
	$(Q)$(MAKE) -C ragas format
	@echo "Formatting experimental..."
	$(Q)$(MAKE) -C experimental format

type: ## Type check all code in the monorepo
	@echo "Type checking all code..."
	@echo "Type checking ragas..."
	$(Q)$(MAKE) -C ragas type
	@echo "Type checking experimental..."
	$(Q)$(MAKE) -C experimental type

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

# =============================================================================
# CI/BUILD
# =============================================================================

run-ci: ## Run complete CI pipeline (mirrors GitHub CI exactly)
	@echo "Running complete CI pipeline..."
	@echo "Running ragas CI..."
	$(Q)$(MAKE) -C ragas run-ci
	@echo "Running experimental CI..."
	$(Q)$(MAKE) -C experimental run-ci
	@echo "All CI checks passed!"

run-ci-format-check: ## Run format check in dry-run mode (like GitHub CI)
	@echo "Running format check (dry-run, like GitHub CI)..."
	@echo "Checking ragas formatting..."
	$(Q)black --check --config ragas/pyproject.toml ragas/src ragas/tests docs
	$(Q)ruff check ragas/src docs ragas/tests
	@echo "Checking experimental formatting..."
	$(Q)cd experimental && black --check ragas_experimental && ruff check ragas_experimental

run-ci-type: ## Run type checking (matches GitHub CI)
	@echo "Running type checking (matches GitHub CI)..."
	$(Q)$(MAKE) type

run-ci-tests: ## Run all tests with GitHub CI options
	@echo "Running unit tests with CI options..."
	$(Q)cd ragas && __RAGAS_DEBUG_TRACKING=true RAGAS_DO_NOT_TRACK=true pytest --nbmake tests/unit --dist loadfile -n auto
	@echo "Running experimental tests with CI options..."
	$(Q)cd experimental && __RAGAS_DEBUG_TRACKING=true RAGAS_DO_NOT_TRACK=true pytest -v --tb=short

run-ci-fast: ## Fast CI check for quick local validation (2-3 minutes)
	@echo "Running fast CI check for quick feedback..."
	@echo "Format check..."
	$(Q)black --check --config ragas/pyproject.toml ragas/src ragas/tests docs
	$(Q)ruff check ragas/src docs ragas/tests
	$(Q)cd experimental && black --check ragas_experimental && ruff check ragas_experimental
	@echo "Core unit tests (no nbmake for speed)..."
	$(Q)cd ragas && pytest tests/unit --dist loadfile -n auto -x
	@echo "Essential experimental tests..."
	$(Q)cd experimental && pytest -v --tb=short -x
	@echo "Fast CI check completed!"

clean: ## Clean all generated files
	@echo "Cleaning all generated files..."
	$(Q)cd $(GIT_ROOT)/docs && $(MAKE) clean
	$(Q)find . -type f -name '*.py[co]' -delete -o -type d -name __pycache__ -delete

# =============================================================================
# TESTING
# =============================================================================

test: ## Run all unit tests in the monorepo
	@echo "Running all unit tests..."
	@echo "Running ragas tests..."
	$(Q)$(MAKE) -C ragas test $(shell if [ -n "$(k)" ]; then echo "k=$(k)"; fi)
	@echo "Running experimental tests..."
	$(Q)$(MAKE) -C experimental test

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
	@echo "Building experimental documentation..."
	$(Q)cd experimental && nbdev_docs

serve-docs: ## Build and serve documentation locally
	$(Q)mkdocs serve --dirtyreload

# Optionally show commands being executed with V=1
Q := $(if $(V),,@)

# Common paths for experimental
EXPERIMENTAL_PATH := ragas_experimental

help: ## Show all Makefile targets
	$(Q)grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[33m%-30s\033[0m %s\n", $$1, $$2}'

# =============================================================================
# CODE QUALITY
# =============================================================================

.PHONY: help format type check test run-ci

format: ## Format and lint experimental code
	@echo "Formatting and linting experimental code..."
	@echo "(black) Formatting experimental..."
	$(Q)uv run black $(EXPERIMENTAL_PATH)
	@echo "(ruff) Auto-fixing experimental (includes import sorting and unused imports)..."
	$(Q)uv run ruff check $(EXPERIMENTAL_PATH) --fix-only
	@echo "(ruff) Final linting check for experimental..."
	$(Q)uv run ruff check $(EXPERIMENTAL_PATH)

type: ## Type check experimental code
	@echo "Type checking experimental code..."
	# TODO: Fix experimental type checking for 0.3 release - currently has 96 type errors
	# $(Q)PYRIGHT_PYTHON_FORCE_VERSION=latest pyright $(EXPERIMENTAL_PATH)
	@echo "Experimental type checking temporarily disabled - TODO: fix for 0.3 release"

check: format type ## Quick health check (format + type, no tests)
	@echo "Experimental code quality check complete!"

test: ## Run experimental unit tests
	@echo "Running experimental unit tests..."
	$(Q)uv run pytest

run-ci: ## Run complete CI pipeline for experimental
	@echo "Running experimental CI pipeline..."
	@echo "Format check..."
	$(Q)uv run black --check $(EXPERIMENTAL_PATH) && uv run ruff check $(EXPERIMENTAL_PATH)
	@echo "Type check..."
	$(Q)$(MAKE) type
	@echo "Unit tests..."
	$(Q)__RAGAS_DEBUG_TRACKING=true RAGAS_DO_NOT_TRACK=true pytest -v --tb=short
	@echo "Experimental CI pipeline complete!"
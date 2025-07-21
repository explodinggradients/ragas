# Optionally show commands being executed with V=1
Q := $(if $(V),,@)

# Common paths for ragas
RAGAS_PATHS := src tests ../docs

help: ## Show all Makefile targets
	$(Q)grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[33m%-30s\033[0m %s\n", $$1, $$2}'

# =============================================================================
# CODE QUALITY
# =============================================================================

.PHONY: help format type check test run-ci

format: ## Format and lint ragas code
	@echo "Formatting and linting ragas code..."
	@echo "(black) Formatting ragas..."
	$(Q)uv run black --config pyproject.toml $(RAGAS_PATHS)
	@echo "(black) Formatting stubs..."
	$(Q)find src -name "*.pyi" ! -name "*_pb2*" -exec uv run black --pyi --config pyproject.toml {} \;
	@echo "(ruff) Auto-fixing ragas (includes import sorting and unused imports)..."
	$(Q)uv run ruff check $(RAGAS_PATHS) --fix-only
	@echo "(ruff) Final linting check for ragas..."
	$(Q)uv run ruff check $(RAGAS_PATHS)

type: ## Type check ragas code
	@echo "Type checking ragas code..."
	@echo "(pyright) Typechecking ragas..."
	$(Q)PYRIGHT_PYTHON_FORCE_VERSION=latest pyright src

check: format type ## Quick health check (format + type, no tests)
	@echo "Ragas code quality check complete!"

test: ## Run ragas unit tests
	@echo "Running ragas unit tests..."
	$(Q)uv run pytest --nbmake tests/unit $(shell if [ -n "$(k)" ]; then echo "-k $(k)"; fi)

run-ci: ## Run complete CI pipeline for ragas
	@echo "Running ragas CI pipeline..."
	@echo "Format check..."
	$(Q)uv run black --check --config pyproject.toml $(RAGAS_PATHS)
	$(Q)uv run ruff check $(RAGAS_PATHS)
	@echo "Type check..."
	$(Q)$(MAKE) type
	@echo "Unit tests..."
	$(Q)__RAGAS_DEBUG_TRACKING=true RAGAS_DO_NOT_TRACK=true pytest --nbmake tests/unit --dist loadfile -n auto
	@echo "Ragas CI pipeline complete!"
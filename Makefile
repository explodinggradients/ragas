GIT_ROOT ?= $(shell git rev-parse --show-toplevel)

# Optionally show commands being executed with V=1
Q := $(if $(V),,@)

help: ## Show all Makefile targets
	$(Q)grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[33m%-30s\033[0m %s\n", $$1, $$2}'

.PHONY: format lint type clean test test-e2e run-ci build-docs serve-docs process-experimental-notebooks
format: ## Format all code in the monorepo
	@echo "Formatting all code..."
	@echo "(isort) Ordering imports in ragas..."
	$(Q)cd ragas && isort .
	@echo "(black) Formatting ragas..."
	$(Q)black --config ragas/pyproject.toml ragas/src ragas/tests docs
	@echo "(black) Formatting stubs..."
	$(Q)find ragas/src -name "*.pyi" ! -name "*_pb2*" -exec black --pyi --config ragas/pyproject.toml {} \;
	@echo "(ruff) Auto-fixing ragas (includes unused imports)..."
	$(Q)ruff check ragas/src docs ragas/tests --fix-only
	@echo "(black) Formatting experimental..."
	$(Q)cd experimental && black ragas_experimental
	@echo "(ruff) Auto-fixing experimental (includes unused imports)..."
	$(Q)cd experimental && ruff check ragas_experimental --fix-only

lint: ## Lint all code in the monorepo
	@echo "Linting all code..."
	@echo "(ruff) Linting ragas..."
	$(Q)ruff check ragas/src docs ragas/tests
	@echo "(ruff) Linting experimental..."
	$(Q)cd experimental && ruff check ragas_experimental

type: ## Type check all code in the monorepo
	@echo "Type checking all code..."
	@echo "(pyright) Typechecking ragas..."
	$(Q)cd ragas && PYRIGHT_PYTHON_FORCE_VERSION=latest pyright src
	@echo "(pyright) Typechecking experimental..."
	$(Q)PYRIGHT_PYTHON_FORCE_VERSION=latest pyright experimental/ragas_experimental
clean: ## Clean all generated files
	@echo "Cleaning all generated files..."
	$(Q)cd $(GIT_ROOT)/docs && $(MAKE) clean
	$(Q)cd $(GIT_ROOT) || exit 1
	$(Q)find . -type f -name '*.py[co]' -delete -o -type d -name __pycache__ -delete

test: ## Run all tests in the monorepo
	@echo "Running all tests..."
	@echo "Running ragas tests..."
	$(Q)cd ragas && pytest --nbmake tests/unit $(shell if [ -n "$(k)" ]; then echo "-k $(k)"; fi)
	@echo "Running experimental tests..."
	$(Q)cd experimental && pytest

test-e2e: ## Run ragas end2end tests
	@echo "Running ragas end2end tests..."
	$(Q)cd ragas && pytest --nbmake tests/e2e -s

run-ci: format lint type test ## Run all CI checks for the monorepo

# Docs
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

serve-docs: ## Build and serve documentation
	$(Q)mkdocs serve --dirtyreload

# Benchmarks
run-benchmarks-eval: ## Run benchmarks for Evaluation
	@echo "Running benchmarks for Evaluation..."
	$(Q)cd $(GIT_ROOT)/ragas/tests/benchmarks && python benchmark_eval.py
run-benchmarks-testset: ## Run benchmarks for TestSet Generation
	@echo "Running benchmarks for TestSet Generation..."
	$(Q)cd $(GIT_ROOT)/ragas/tests/benchmarks && python benchmark_testsetgen.py
run-benchmarks-in-docker: ## Run benchmarks in docker
	@echo "Running benchmarks in docker..."
	$(Q)cd $(GIT_ROOT)
	docker buildx build --build-arg OPENAI_API_KEY=$(OPENAI_API_KEY) -t ragas-benchmark -f $(GIT_ROOT)/ragas/tests/benchmarks/Dockerfile .
	docker inspect ragas-benchmark:latest | jq ".[0].Size" | numfmt --to=si

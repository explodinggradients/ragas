GIT_ROOT ?= $(shell git rev-parse --show-toplevel)

# Optionally show commands being executed with V=1
Q := $(if $(V),,@)

help: ## Show all Makefile targets
	$(Q)grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[33m%-30s\033[0m %s\n", $$1, $$2}'

.PHONY: format lint type style clean run-benchmarks format-experimental lint-experimental type-experimental process-experimental-notebooks
format: ## Running code formatter for ragas
	@echo "(isort) Ordering imports..."
	$(Q)cd ragas && isort .
	@echo "(black) Formatting codebase..."
	$(Q)black --config ragas/pyproject.toml ragas/src ragas/tests docs
	@echo "(black) Formatting stubs..."
	$(Q)find ragas/src -name "*.pyi" ! -name "*_pb2*" -exec black --pyi --config ragas/pyproject.toml {} \;
	@echo "(ruff) Running fix only..."
	$(Q)ruff check ragas/src docs ragas/tests --fix-only

format-experimental: ## Running code formatter for experimental
	@echo "(black) Formatting experimental codebase..."
	$(Q)cd experimental && black ragas_experimental
	@echo "(ruff) Running fix only on experimental..."
	$(Q)ruff check experimental/ragas_experimental --fix-only

format-all: format format-experimental ## Format all code in the monorepo

lint: ## Running lint checker for ragas
	@echo "(ruff) Linting ragas project..."
	$(Q)ruff check ragas/src docs ragas/tests

lint-experimental: ## Running lint checker for experimental
	@echo "(ruff) Linting experimental project..."
	$(Q)ruff check experimental/ragas_experimental

lint-all: lint lint-experimental ## Lint all code in the monorepo

type: ## Running type checker for ragas
	@echo "(pyright) Typechecking ragas codebase..."
	PYRIGHT_PYTHON_FORCE_VERSION=latest pyright ragas/src/ragas

type-experimental: ## Running type checker for experimental
	@echo "(pyright) Typechecking experimental codebase..."
	PYRIGHT_PYTHON_FORCE_VERSION=latest pyright experimental/ragas_experimental

type-all: type type-experimental ## Type check all code in the monorepo
clean: ## Clean all generated files
	@echo "Cleaning all generated files..."
	$(Q)cd $(GIT_ROOT)/docs && $(MAKE) clean
	$(Q)cd $(GIT_ROOT) || exit 1
	$(Q)find . -type f -name '*.py[co]' -delete -o -type d -name __pycache__ -delete

test: ## Run ragas tests
	@echo "Running ragas tests..."
	$(Q)cd ragas && pytest --nbmake tests/unit $(shell if [ -n "$(k)" ]; then echo "-k $(k)"; fi)

test-e2e: ## Run ragas end2end tests
	echo "running ragas end2end tests..."
	$(Q)cd ragas && pytest --nbmake tests/e2e -s

test-experimental: ## Run experimental tests
	@echo "Running experimental tests..."
	$(Q)cd experimental && pytest

test-all: test test-experimental ## Run all tests

run-ci: format lint type test ## Running all CI checks for ragas

run-ci-experimental: format-experimental lint-experimental type-experimental test-experimental ## Running all CI checks for experimental

run-ci-all: format-all lint-all type-all test-all ## Running all CI checks for both projects

# Docs
build-docsite-ragas: ## Build ragas documentation
	@echo "convert ipynb notebooks to md files"
	$(Q)python $(GIT_ROOT)/docs/ipynb_to_md.py
	$(Q)mkdocs build

process-experimental-notebooks: ## Process experimental notebooks to markdown for MkDocs
	@echo "Processing experimental notebooks..."
	$(Q)python $(GIT_ROOT)/scripts/process_experimental_notebooks.py

build-docsite-experimental: process-experimental-notebooks ## Build experimental documentation
	@echo "Building experimental documentation..."
	$(Q)cd experimental && nbdev_docs

build-docsite: build-docsite-ragas ## Build all documentation

serve-docsite: ## Build and serve documentation
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

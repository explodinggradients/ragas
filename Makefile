GIT_ROOT ?= $(shell git rev-parse --show-toplevel)

# Optionally show commands being executed with V=1
Q := $(if $(V),,@)

help: ## Show all Makefile targets
	$(Q)grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[33m%-30s\033[0m %s\n", $$1, $$2}'

.PHONY: format lint type style clean run-benchmarks
format: ## Running code formatter: black and isort
	@echo "(isort) Ordering imports..."
	$(Q)isort .
	@echo "(black) Formatting codebase..."
	$(Q)black --config pyproject.toml src tests docs
	@echo "(black) Formatting stubs..."
	$(Q)find src -name "*.pyi" ! -name "*_pb2*" -exec black --pyi --config pyproject.toml {} \;
	@echo "(ruff) Running fix only..."
	$(Q)ruff check src docs tests --fix-only
lint: ## Running lint checker: ruff
	@echo "(ruff) Linting development project..."
	$(Q)ruff check src docs tests
type: ## Running type checker: pyright
	@echo "(pyright) Typechecking codebase..."
	PYRIGHT_PYTHON_FORCE_VERSION=latest pyright src/ragas
clean: ## Clean all generated files
	@echo "Cleaning all generated files..."
	$(Q)cd $(GIT_ROOT)/docs && $(MAKE) clean
	$(Q)cd $(GIT_ROOT) || exit 1
	$(Q)find . -type f -name '*.py[co]' -delete -o -type d -name __pycache__ -delete
test: ## Run tests
	@echo "Running tests..."
	$(Q)pytest --nbmake tests/unit $(shell if [ -n "$(k)" ]; then echo "-k $(k)"; fi)
test-e2e: ## Run end2end tests
	echo "running end2end tests..."
	$(Q)pytest --nbmake tests/e2e -s
run-ci: format lint type test ## Running all CI checks

# Docs
rewrite-docs: ## Use GPT4 to rewrite the documentation
	@echo "Rewriting the documentation in directory $(DIR)..."
	$(Q)python $(GIT_ROOT)/docs/python alphred.py --directory $(DIR)
docsite: ## Build and serve documentation
	$(Q)mkdocs serve --dirtyreload

# Benchmarks
run-benchmarks-eval: ## Run benchmarks for Evaluation
	@echo "Running benchmarks for Evaluation..."
	$(Q)cd $(GIT_ROOT)/tests/benchmarks && python benchmark_eval.py
run-benchmarks-testset: ## Run benchmarks for TestSet Generation
	@echo "Running benchmarks for TestSet Generation..."
	$(Q)cd $(GIT_ROOT)/tests/benchmarks && python benchmark_testsetgen.py
run-benchmarks-in-docker: ## Run benchmarks in docker
	@echo "Running benchmarks in docker..."
	$(Q)cd $(GIT_ROOT)
	docker buildx build --build-arg OPENAI_API_KEY=$(OPENAI_API_KEY) -t ragas-benchmark -f $(GIT_ROOT)/tests/benchmarks/Dockerfile .
	docker inspect ragas-benchmark:latest | jq ".[0].Size" | numfmt --to=si

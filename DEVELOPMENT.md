# Development Guide for Ragas Monorepo

This comprehensive guide covers development workflows for the Ragas monorepo, designed for both human developers and AI agents.

## Quick Start (for Developers)

```bash
# 1. Clone and enter the repository
git clone https://github.com/explodinggradients/ragas.git
cd ragas

# 2. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Set up development environment
make setup

# 4. Verify everything works
make check

# 5. Start developing!
make help  # See all available commands
```

## Quick Start (for AI Agents)

AI agents working with this codebase should use these standardized commands:

```bash
# Essential commands for AI development
make help           # See all available targets
make setup          # Complete environment setup
make check          # Quick health check (format + type)
make test           # Run all tests
make run-ci         # Full CI pipeline locally

# Individual development tasks
make format         # Format and lint all code
make type           # Type check all code
make clean          # Clean generated files
```

**Key Points for AI Agents:**
- Always use `make` commands rather than direct tool invocation
- Use `uv run` prefix for any direct Python tool usage
- Check `make help` for the complete command reference
- The CI pipeline uses the same commands as local development

## Monorepo Architecture

This repository is organized as a monorepo containing two main projects:

```
ragas/
├── ragas/                          # Core evaluation library
│   ├── src/ragas/                 # Main source code
│   ├── tests/                     # Tests (unit, e2e, benchmarks)
│   └── pyproject.toml             # Dependencies and configuration
│
├── experimental/                   # Experimental extensions
│   ├── ragas_experimental/        # Generated Python code
│   ├── tests/                     # Pytest-based tests
│   └── pyproject.toml             # Dependencies and configuration
│
├── docs/                          # Combined documentation
├── .github/workflows/             # CI/CD pipeline
├── Makefile                       # Unified build commands
└── CLAUDE.md                      # AI assistant instructions
```

### Project Relationships
- **Ragas Core**: The main evaluation toolkit for LLM applications
- **Ragas Experimental**: Extensions for advanced features and UI components
- **Shared Infrastructure**: Unified CI/CD, documentation, and build system

## Development Environment Setup

### Prerequisites
- Python 3.9+ 
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- Git

### Setup Process

#### Option 1: Using Make (Recommended)
```bash
make setup
```

#### Option 2: Manual Setup
```bash
# Install uv if not available
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install both projects
uv pip install -e "./ragas[dev]"
uv pip install -e "./experimental[dev]"
```

### Verification
```bash
make check  # Runs format + type checking
make test   # Runs all tests
```

## Available Commands Reference

Run `make help` to see all targets. Here are the essential commands:

### Setup & Installation
- `make install` - Install dependencies for both projects
- `make setup` - Complete development environment setup

### Code Quality
- `make format` - Format and lint all code (includes unused import cleanup)
- `make type` - Type check all code
- `make check` - Quick health check (format + type, no tests)

### Testing
- `make test` - Run all unit tests
- `make test-e2e` - Run end-to-end tests
- `make benchmarks` - Run performance benchmarks
- `make benchmarks-docker` - Run benchmarks in Docker

### CI/Build
- `make run-ci` - Run complete CI pipeline locally
- `make clean` - Clean all generated files

### Documentation
- `make build-docs` - Build all documentation
- `make serve-docs` - Serve documentation locally

## Development Workflows

### Daily Development
```bash
# 1. Start your work
git checkout -b feature/your-feature

# 2. Make changes to code

# 3. Check your work
make check           # Format and type check
make test            # Run tests

# 4. Commit and push
git add .
git commit -m "feat: your feature description"
git push origin feature/your-feature
```

### Before Submitting PR
```bash
make run-ci          # Run full CI pipeline
# Ensure all checks pass before creating PR
```

### Working with Specific Projects

#### Ragas Core Development
```bash
# Navigate to the ragas directory for project-specific work
cd ragas
uv run pytest tests/unit          # Run specific tests
uv run pyright src               # Type check specific code
```

#### Experimental Development
```bash
# Navigate to experimental directory
cd experimental  
uv run pytest                   # Run experimental tests
```

## Testing Strategy

### Test Types
1. **Unit Tests**: Fast, isolated tests for individual components
2. **End-to-End Tests**: Integration tests for complete workflows
3. **Benchmarks**: Performance tests for evaluation metrics

### Running Tests
```bash
# All tests
make test

# Specific test categories
cd ragas && uv run pytest tests/unit
cd ragas && uv run pytest tests/e2e  
cd experimental && uv run pytest

# With coverage or specific options
cd ragas && uv run pytest tests/unit -k "test_name"
```

### Test Organization
- **Ragas Core**: `ragas/tests/` (unit, e2e, benchmarks)
- **Experimental**: `experimental/tests/` (unit, e2e)

## Code Quality & CI/CD

### Code Quality Pipeline
The `make format` command runs:
1. **isort**: Import sorting
2. **black**: Code formatting
3. **ruff --fix-only**: Auto-fix issues (including unused imports)
4. **ruff check**: Final linting validation

### Type Checking
```bash
make type  # Type check all code with pyright
```

### CI/CD Pipeline
Our GitHub Actions CI runs:
1. **Dependency Installation**: Using uv for consistent environments
2. **Code Quality Checks**: Format and type validation
3. **Testing**: Unit and integration tests across Python 3.9-3.12
4. **Multi-OS Testing**: Ubuntu, macOS, Windows

### Local CI Simulation
```bash
make run-ci  # Runs: format + type + test
```

## Project-Specific Guidelines

### Ragas Core
- **Language**: Python with type hints
- **Testing**: pytest with nbmake for notebook tests
- **Style**: Google-style docstrings
- **Architecture**: Modular metrics and evaluation framework

### Experimental
- **Dependencies**: Defined in `pyproject.toml`
- **Testing**: Pure pytest (no nbdev)
- **Features**: Advanced evaluation tools and UI components

### Adding Dependencies
- **Ragas Core**: Add to `ragas/pyproject.toml`
- **Experimental**: Add to `experimental/pyproject.toml`
- **Always**: Test with `make install` and `make test`

## Troubleshooting

### Common Issues

#### Import Errors
```bash
# Reinstall in development mode
make install
```

#### Test Failures
```bash
# Run specific failing test
cd ragas && uv run pytest tests/unit/test_specific.py -v

# Check test dependencies
cd experimental && uv run pytest --collect-only
```

#### Formatting Issues
```bash
# Fix formatting
make format

# Check specific files
uv run ruff check path/to/file.py --fix
```

#### CI Failures
```bash
# Run the same checks locally
make run-ci

# Individual checks
make format  # Must pass
make type    # Must pass  
make test    # Must pass
```

### Development Environment Issues

#### uv Not Found
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
# or use pip: pip install uv
```

#### Dependency Conflicts
```bash
# Clean install
make clean
make install
```

### Getting Help
- **Documentation**: Check `CLAUDE.md` for AI assistant guidance
- **Commands**: Run `make help` for all available targets
- **Issues**: Check existing GitHub issues or create a new one

## Contributing Guidelines

### Pull Request Process
1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Develop** using the workflows above
4. **Test** thoroughly: `make run-ci`
5. **Submit** a pull request with clear description

### Commit Message Format
```
feat: add new evaluation metric
fix: resolve import error in experimental
docs: update development guide
test: add unit tests for metric base
```

### Code Review Checklist
- [ ] All tests pass (`make test`)
- [ ] Code is formatted (`make format`)
- [ ] Type checking passes (`make type`)
- [ ] Documentation is updated
- [ ] Appropriate tests are included

## AI Agent Best Practices

### Recommended Workflow for AI Agents
1. **Understand the task**: Read relevant documentation and code
2. **Plan the approach**: Identify which project(s) need changes
3. **Use standardized commands**: Always prefer `make` targets
4. **Test incrementally**: Use `make check` frequently during development
5. **Validate thoroughly**: Run `make run-ci` before completing

### Command Patterns for AI Agents
```bash
# Always start with understanding the current state
make help
ls -la  # Check current directory structure

# For code changes
make format  # After making changes
make test    # Verify functionality

# For investigation
uv run pytest --collect-only  # See available tests
uv run ruff check --no-fix    # Check issues without fixing
```

### File Modification Guidelines
- **Prefer editing** existing files over creating new ones
- **Use project conventions** (check similar files for patterns)
- **Update tests** when modifying functionality
- **Follow existing code style** (enforced by `make format`)

---

**Happy coding! 🚀**

For additional context and instructions specific to AI assistants, see [CLAUDE.md](./CLAUDE.md).
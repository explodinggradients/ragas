# Development Guide for ragas

This document provides guidelines for developing and contributing to the ragas project.

## Setting up the Development Environment

1. **Fork the Repository**
   Fork the [ragas repository](https://github.com/explodinggradients/ragas) on GitHub.

2. **Clone your Fork**
   ```
   git clone https://github.com/YOUR_USERNAME/ragas.git
   cd ragas
   ```

3. **Set up a Virtual Environment**
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

4. **Install Dependencies**
   ```
   pip install -e ".[dev]"
   ```

## Development Workflow

1. **Create a New Branch**
   ```
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes and Commit**
   ```
   git add .
   git commit -m "Your descriptive commit message"
   ```

3. **Push Changes to Your Fork**
   ```
   git push origin feature/your-feature-name
   ```

4. **Create a Pull Request**
   Go to the original ragas repository and create a new pull request from your feature branch.

## Coding Standards

- Follow PEP 8 guidelines for Python code.
- Use type hints where possible.
- Write docstrings for all functions, classes, and modules.
- Ensure all tests pass before submitting a pull request.

You can run the following command to check for code style issues:
```bash
make run-ci
```

## Running Tests

To run the test suite:

```bash
make test
```

## Documentation

- Update documentation for any new features or changes to existing functionality.
- Use [Google style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) for docstrings.

## Submitting Pull Requests

1. Ensure your code adheres to the project's coding standards.
2. Include tests for new functionality.
3. Update documentation as necessary.
4. Provide a clear description of the changes in your pull request.

Thank you for contributing to ragas!

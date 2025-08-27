# Ragas Examples

Official examples demonstrating how to use Ragas for evaluating different types of AI applications including RAG systems, agents, prompts, workflows, and LLM benchmarking. These examples might be unstable and are subject to change.

## Installation

### From PyPI (after release)
```bash
pip install "ragas[examples]"
```

### Local Development
Install both main ragas and examples packages in editable mode:

```bash
cd /path/to/ragas
uv pip install -e . -e ./examples
```

Or using regular pip:
```bash
cd /path/to/ragas  
pip install -e . -e ./examples
```

## Available Examples

- **`ragas_examples.agent_evals`** - Agent evaluation examples
- **`ragas_examples.benchmark_llm`** - LLM benchmarking and comparison examples  
- **`ragas_examples.prompt_evals`** - Prompt evaluation examples
- **`ragas_examples.rag_eval`** - RAG system evaluation examples
- **`ragas_examples.workflow_eval`** - Workflow evaluation examples

## Usage

### Set Environment Variables

Most examples require API keys to be set:

```bash
export OPENAI_API_KEY=your_key_here
```

For Google Drive examples, also install the gdrive extra:
```bash
pip install "ragas[examples,gdrive]"
```

### Running Examples as Modules

After installation, you can run examples directly:

```bash
# Run benchmark LLM prompt example
python -m ragas_examples.benchmark_llm.prompt

# Run benchmark LLM evaluation
python -m ragas_examples.benchmark_llm.evals

# Run other examples
python -m ragas_examples.rag_eval.evals
python -m ragas_examples.agent_evals.evals
python -m ragas_examples.prompt_evals.evals
python -m ragas_examples.workflow_eval.evals
```

## Release process

- The examples package is versioned independently using Git tags with prefix `examples-v` (e.g., `examples-v0.1.0`).
- Publishing is handled by the GitHub Actions workflow `publish-examples.yml`, which builds from `examples/` and publishes to PyPI when such a tag is pushed.

### Release Commands

To create and push a new release:

```bash
# Create and push a new tag (replace X.Y.Z with actual version)
git tag examples-vX.Y.Z
git push origin examples-vX.Y.Z

# Example:
git tag examples-v0.1.0
git push origin examples-v0.1.0
```

## Local Development & Testing


## Local Development & Testing

### Verify Installation
```bash

# Test module execution
python -m ragas_examples.benchmark_llm.prompt --help
```

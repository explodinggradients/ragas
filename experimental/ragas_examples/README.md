# Ragas Examples

This package contains comprehensive examples demonstrating how to use Ragas for evaluating different types of AI applications including RAG systems, agents, prompts, and workflows.

## Installation

Install the ragas_experimental package with examples dependencies:

```bash
pip install -e ".[examples]"
```

## Usage

### Command Line Interface

Run examples using the CLI:

```bash
# Run a specific example
python -m ragas_examples rag
python -m ragas_examples agent
python -m ragas_examples prompt  
python -m ragas_examples workflow

# Or using the console script (after installation)
ragas-examples rag
ragas-examples agent

# List all available examples
python -m ragas_examples --list
ragas-examples --list
```

### Programmatic Usage

Import and use examples in your code:

```python
from ragas_examples import (
    default_rag_client,
    get_default_agent,
    run_prompt,
    default_workflow_client
)

# Use RAG client
rag_client = default_rag_client(llm_client=your_llm_client)
response = rag_client.query("What is ragas?")

# Use math agent
agent = get_default_agent()
result = agent.solve("2 + 3 * 4")

# Use prompt classifier
classification = run_prompt("This movie was amazing!")

# Use workflow client
workflow_client = default_workflow_client()
result = workflow_client.process_email(email_content)
```

## Examples

### 1. RAG Evaluation (`rag`)

**Location**: `ragas_examples/rag_eval/`

A complete RAG (Retrieval-Augmented Generation) system with:
- Document retrieval using keyword matching
- Response generation using OpenAI
- Comprehensive tracing and logging
- Evaluation using discrete metrics

**Features**:
- Configurable retriever (SimpleKeywordRetriever included)
- Automatic trace logging to JSON files
- Error handling and recovery
- Evaluation against grading notes

**Run**: `python -m ragas_examples rag`

### 2. Agent Evaluation (`agent`)

**Location**: `ragas_examples/agent_evals/`

A mathematical agent that solves complex expressions using atomic operations:
- Function calling with add, subtract, multiply, divide
- Step-by-step problem decomposition
- Iterative planning with LLM
- Numerical correctness evaluation

**Features**:
- Atomic mathematical operations
- Trace logging for each computation step
- Error handling for invalid operations
- Correctness metric evaluation

**Run**: `python -m ragas_examples agent`

### 3. Prompt Evaluation (`prompt`)

**Location**: `ragas_examples/prompt_evals/`

A simple prompt-based sentiment classification system:
- Movie review sentiment analysis
- Binary classification (positive/negative)
- Accuracy evaluation
- Discrete metric scoring

**Features**:
- Simple prompt-based classification
- Accuracy metrics
- Dataset loading and evaluation
- Pass/fail scoring

**Run**: `python -m ragas_examples prompt`

### 4. Workflow Evaluation (`workflow`)

**Location**: `ragas_examples/workflow_eval/`

A support ticket triage workflow with configurable extraction:
- Email classification (Bug Report, Billing, Feature Request)
- Configurable information extraction (regex vs LLM)
- Automated response generation
- Multi-criteria evaluation

**Features**:
- Configurable extraction modes (deterministic/LLM)
- Multi-step workflow processing
- Response quality evaluation
- Comprehensive tracing

**Run**: `python -m ragas_examples workflow`

## Environment Setup

All examples require an OpenAI API key:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

## File Structure

```
ragas_examples/
├── __init__.py                 # Package exports
├── __main__.py                 # CLI interface
├── README.md                   # This file
├── rag_eval/
│   ├── __init__.py
│   ├── rag.py                  # RAG system implementation
│   └── evals.py                # RAG evaluation logic
├── agent_evals/
│   ├── __init__.py
│   ├── agent.py                # Mathematical agent implementation
│   └── evals.py                # Agent evaluation logic
├── prompt_evals/
│   ├── __init__.py
│   ├── prompt.py               # Prompt classification implementation
│   └── evals.py                # Prompt evaluation logic
└── workflow_eval/
    ├── __init__.py
    ├── workflow.py             # Workflow implementation
    └── evals.py                # Workflow evaluation logic
```

## Development

To extend or modify examples:

1. Each example has an implementation file and evaluation file
2. Add new examples by creating a new directory with `__init__.py`, implementation, and evaluation files
3. Update `__main__.py` to include new examples in the CLI
4. Update `__init__.py` to export new functionality
5. Add dependencies to `pyproject.toml` if needed

## Output

Each example generates:
- Console output showing evaluation progress
- JSON trace files in a `logs/` directory
- Evaluation results with metrics and scores
- Experiment datasets saved locally

## Troubleshooting

**Import errors**: Ensure you've installed with `pip install -e ".[examples]"`

**API errors**: Check your OpenAI API key is set correctly

**Permission errors**: Ensure you have write permissions for log directories

**Module not found**: Run examples from the experimental directory or install the package properly
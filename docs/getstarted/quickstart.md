# Quick Start: Get Evaluations Running in a flash

Get started with Ragas in seconds. No installation needed! Just set your API key and run one command.

## 1. Set Your API Key

Choose your LLM provider:

```sh
# OpenAI (default)
export OPENAI_API_KEY="your-openai-key"

# Or use Anthropic Claude
export ANTHROPIC_API_KEY="your-anthropic-key"
```

## 2. Create Your Project

Create a complete project with a single command using `uvx` (no installation required):

```sh
uvx ragas quickstart rag_eval
cd rag_eval
```

That's it! You now have a fully configured evaluation project ready to use.

## Project Structure

Your generated project includes:

```sh
rag_eval/
├── README.md              # Project documentation
├── evals.py               # Evaluation configuration
├── rag.py                 # Your LLM application
└── evals/
    ├── datasets/          # Test data (CSV/JSON files)
    ├── experiments/       # Evaluation results
    └── logs/              # Execution logs
```

## Run Evaluations

### Run the Evaluation

Execute the evaluation on your dataset:

```sh
uvx ragas evals evals.py --dataset test_data --metrics faithfulness,answer_correctness
```

Or, if you prefer to use Python directly (after installing ragas):

```sh
python evals.py
```

This will:
- Load test data from `evals/datasets/`
- Evaluate your application using pre-configured metrics
- Save results to `evals/experiments/`

### View Results

Results are saved as CSV files in `evals/experiments/`:

```python
import pandas as pd

# Load and view results
df = pd.read_csv('evals/experiments/results.csv')
print(df[['user_input', 'response', 'faithfulness', 'answer_correctness']])

# Quick statistics
print(f"Average Faithfulness: {df['faithfulness'].mean():.2f}")
print(f"Average Correctness: {df['answer_correctness'].mean():.2f}")
```

...

## Customize Your Evaluation

### Add More Test Cases

Edit `evals/datasets/test_data.csv`:

```csv
user_input,response,reference
What is Ragas?,Ragas is an evaluation framework for LLM applications,Ragas provides objective metrics for evaluating LLM applications
How do metrics work?,Metrics score your LLM outputs,Metrics evaluate the quality and performance of LLM responses
```

### Change the LLM Provider

In `evals.py`, update the LLM configuration:

```python
from ragas.llms import llm_factory

# Use Anthropic Claude
llm = llm_factory("claude-3-5-sonnet-20241022", provider="anthropic")

# Use Google Gemini
llm = llm_factory("gemini-1.5-pro", provider="google")

# Use local Ollama
llm = llm_factory("mistral", provider="ollama", base_url="http://localhost:11434")
```

### Select Different Metrics

In `evals.py`, modify the metrics list:

```python
from ragas.metrics import (
    Faithfulness,           # Does response match context?
    AnswerCorrectness,      # Is the answer correct?
    ContextPrecision,       # Is retrieved context relevant?
    ContextRecall,          # Is all needed context retrieved?
)

# Use only specific metrics
metrics = [
    Faithfulness(),
    AnswerCorrectness(),
]
```

## What's Next?

- **Learn the concepts**: Read the [Evaluate a Simple LLM Application](evals.md) guide for deeper understanding
- **Custom metrics**: [Write your own metrics](../howtos/customizations/metrics/_write_your_own_metric.md) tailored to your use case
- **Production integration**: [Integrate evaluations into your CI/CD pipeline](../howtos/index.md)
- **RAG evaluation**: Evaluate [RAG systems](rag_eval.md) with specialized metrics
- **Agent evaluation**: Explore [AI agent evaluation](../howtos/applications/text2sql.md)
- **Test data generation**: [Generate synthetic test datasets](rag_testset_generation.md) for your evaluations

## Getting Help

- 📚 [Full Documentation](https://docs.ragas.io/)
- 💬 [Join our Discord Community](https://discord.gg/5djav8GGNZ)
- 🐛 [Report Issues](https://github.com/explodinggradients/ragas/issues)

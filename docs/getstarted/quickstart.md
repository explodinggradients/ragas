# Quick Start: Get Evaluations Running in a Flash

Get started with Ragas in minutes. Create a complete evaluation project with just a few commands.

## Step 1: Create Your Project

Choose one of the following methods:

=== "uvx (Recommended)"
    No installation required. `uvx` automatically downloads and runs ragas:

    ```sh
    uvx ragas quickstart rag_eval
    cd rag_eval
    ```

=== "Install Ragas First"
    Install ragas first, then create the project:

    ```sh
    pip install ragas
    ragas quickstart rag_eval
    cd rag_eval
    ```

## Step 2: Install Dependencies

Install the project dependencies:

```sh
uv sync
```

Or if you prefer `pip`:

```sh
pip install -e .
```

## Step 3: Set Your API Key

Choose your LLM provider and set the environment variable:

```sh
# OpenAI (default)
export OPENAI_API_KEY="your-openai-key"

# Or use Anthropic Claude
export ANTHROPIC_API_KEY="your-anthropic-key"

# Or use Google Gemini
export GOOGLE_API_KEY="your-google-key"
```

## Project Structure

Your generated project includes:

```sh
rag_eval/
├── README.md              # Project documentation
├── pyproject.toml         # Project configuration
├── rag.py                 # Your RAG application
├── evals.py               # Evaluation workflow
├── __init__.py            # Makes this a Python package
└── evals/
    ├── datasets/          # Test data files
    ├── experiments/       # Evaluation results
    └── logs/              # Execution logs
```

## Step 4: Run Your Evaluation

Run the evaluation script:

```sh
uv run python evals.py
```

Or if you installed with `pip`:

```sh
python evals.py
```

The evaluation will:
- Load test data from the `load_dataset()` function in `evals.py`
- Query your RAG application with test questions
- Evaluate responses
- Display results in the console
- Save results to CSV in the `evals/experiments/` directory

![](../_static/imgs/results/rag_eval_result.png)

---

## Customize Your Evaluation

### Add More Test Cases

Edit the `load_dataset()` function in `evals.py` to add more test questions:

```python
from ragas.dataset_schema import SingleTurnSample

def load_dataset():
    """Load test dataset for evaluation."""
    data_samples = [
        SingleTurnSample(
            user_input="What is Ragas?",
            response="",  # Will be filled by querying RAG
            reference="Ragas is an evaluation framework for LLM applications",
            retrieved_contexts=[],
        ),
        SingleTurnSample(
            user_input="How do metrics work?",
            response="",
            reference="Metrics evaluate the quality and performance of LLM responses",
            retrieved_contexts=[],
        ),
        # Add more test cases here
    ]

    dataset = EvaluationDataset(samples=data_samples)
    return dataset
```

### Change the LLM Provider

In the `_init_clients()` function in `evals.py`, update the LLM factory call:

```python
from ragas.llms import llm_factory

def _init_clients():
    """Initialize OpenAI client and RAG system."""
    openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    rag_client = default_rag_client(llm_client=openai_client)

    # Use Anthropic Claude instead
    llm = llm_factory("claude-3-5-sonnet-20241022", provider="anthropic")

    # Or use Google Gemini
    # llm = llm_factory("gemini-1.5-pro", provider="google")

    # Or use local Ollama
    # llm = llm_factory("mistral", provider="ollama", base_url="http://localhost:11434")

    return openai_client, rag_client, llm
```

### Customize Dataset and RAG System

The template includes:
- `load_dataset()` - Define your test cases with `SingleTurnSample`
- `query_rag_system()` - Connect to your RAG system
- `evaluate_dataset()` - Implement your evaluation logic
- `display_results()` - Show results in the console
- `save_results_to_csv()` - Export results to CSV

Edit these functions to customize your evaluation workflow.

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

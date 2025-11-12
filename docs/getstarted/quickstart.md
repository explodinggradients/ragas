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

By default, the quickstart example uses OpenAI. Set your API key and you're ready to go. You can also use some other provider with a minor change:

=== "OpenAI (Default)"
    ```sh
    export OPENAI_API_KEY="your-openai-key"
    ```

    The quickstart project is already configured to use OpenAI. You're all set!

=== "Anthropic Claude"
    Set your Anthropic API key:

    ```sh
    export ANTHROPIC_API_KEY="your-anthropic-key"
    ```

    Then update the `_init_clients()` function in `evals.py`:

    ```python
    from ragas.llms import llm_factory

    llm = llm_factory("claude-3-5-sonnet-20241022", provider="anthropic")
    ```

=== "Google Gemini"
    Set up your Google credentials:

    ```sh
    export GOOGLE_API_KEY="your-google-api-key"
    ```

    Then update the `_init_clients()` function in `evals.py`:

    ```python
    from ragas.llms import llm_factory

    llm = llm_factory("gemini-1.5-pro", provider="google")
    ```

=== "Local Models (Ollama)"
    Install and run Ollama locally, then update the `_init_clients()` function in `evals.py`:

    ```python
    from ragas.llms import llm_factory

    llm = llm_factory(
        "mistral",
        provider="ollama",
        base_url="http://localhost:11434"  # Default Ollama URL
    )
    ```

=== "Custom / Other Providers"
    For any LLM with OpenAI-compatible API:

    ```python
    from ragas.llms import llm_factory

    llm = llm_factory(
        "model-name",
        api_key="your-api-key",
        base_url="https://your-api-endpoint"
    )
    ```

    For more details, learn about [LLM integrations](../concepts/metrics/index.md).

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

Congratulations! You have a complete evaluation setup running. 🎉

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

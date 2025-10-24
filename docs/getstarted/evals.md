# Evaluate a simple LLM application

The purpose of this guide is to illustrate a simple workflow for testing and evaluating an LLM application with `ragas`. It assumes minimum knowledge in AI application building and evaluation. Please refer to our [installation instruction](./install.md) for installing `ragas`

!!! tip "Get a Working Example"
    The fastest way to see these concepts in action is to create a project using the quickstart command:

    ```sh
    ragas quickstart rag_eval
    ```

    This generates a complete project with sample code. Follow along with this guide to understand what's happening in your generated code.

    ```sh
    cd rag_eval
    ```

    Let's get started

## Project Structure

Here's what gets created for you:

```sh
rag_eval/
├── README.md             # Quick start guide for your project
├── evals.py              # Your evaluation code (metrics + datasets)
├── rag.py                # Your RAG/LLM application
└── evals/                # Evaluation artifacts
    ├── datasets/         # Test data files (edit these to add more test cases)
    ├── experiments/      # Results from running evaluations
    └── logs/             # Evaluation execution logs
```

**Key files to focus on:**

- **`evals.py`** - Where you define metrics and load test data (we'll explore this next)
- **`rag.py`** - Your application code (query engine, retrieval, etc.)
- **`evals/datasets/`** - Add your test cases here as CSV or JSON files

## Understanding the Code

In your generated project's `evals.py` file, you'll see two key patterns for evaluation:

1. **Metrics** - Functions that score your application's output
2. **Datasets** - Test data that your application is evaluated against

`ragas` offers a variety of evaluation methods, referred to as [metrics](../concepts/metrics/available_metrics/index.md). Let's walk through the most common ones you'll encounter.

### Custom Evaluation with LLMs

In your generated project, you'll see the `DiscreteMetric` - a flexible metric that uses an LLM to evaluate based on any criteria you define:

```python
from ragas.metrics import DiscreteMetric
from ragas.llms import instructor_llm_factory

# Create your evaluator LLM
evaluator_llm = instructor_llm_factory("openai", model="gpt-4o")

# Define a custom metric
my_metric = DiscreteMetric(
    name="correctness",
    prompt="Check if the response is correct. Return 'pass' or 'fail'.\nResponse: {response}\nExpected: {expected}",
    allowed_values=["pass", "fail"],
)

# Use it to score
score = my_metric.score(
    llm=evaluator_llm,
    response="The capital of France is Paris",
    expected="Paris"
)
print(f"Score: {score.value}")  # Output: 'pass'
```

What you see in your generated `evals.py` lets you define evaluation logic that matters for your application. Learn more about [custom metrics](../concepts/metrics/index.md).

### Choosing Your Evaluator LLM

Your evaluation metrics need an LLM to score your application. Ragas works with **any LLM provider** through the `instructor_llm_factory`. Your quickstart project uses OpenAI by default, but you can easily swap to any provider by updating the LLM creation in your `evals.py`:

=== "OpenAI"
    Set your OpenAI API key:

    ```sh
    export OPENAI_API_KEY="your-openai-key"
    ```

    In your `evals.py`:

    ```python
    from ragas.llms import instructor_llm_factory

    llm = instructor_llm_factory("openai", model="gpt-4o")
    ```

    The quickstart project already sets this up for you!

=== "Anthropic Claude"
    Set your Anthropic API key:

    ```sh
    export ANTHROPIC_API_KEY="your-anthropic-key"
    ```

    In your `evals.py`:

    ```python
    from ragas.llms import instructor_llm_factory

    llm = instructor_llm_factory("anthropic", model="claude-3-5-sonnet-20241022")
    ```

=== "Google Cloud"
    Set up your Google credentials:

    ```sh
    export GOOGLE_API_KEY="your-google-api-key"
    ```

    In your `evals.py`:

    ```python
    from ragas.llms import instructor_llm_factory

    llm = instructor_llm_factory("google", model="gemini-1.5-pro")
    ```

=== "Local Models (Ollama)"
    Install and run Ollama locally, then in your `evals.py`:

    ```python
    from ragas.llms import instructor_llm_factory

    llm = instructor_llm_factory(
        "ollama",
        model="mistral",
        base_url="http://localhost:11434"  # Default Ollama URL
    )
    ```

=== "Custom / Other Providers"
    For any LLM with OpenAI-compatible API:

    ```python
    from ragas.llms import instructor_llm_factory

    llm = instructor_llm_factory(
        "openai",
        api_key="your-api-key",
        base_url="https://your-api-endpoint"
    )
    ```

    For more details, learn about [LLM integrations](../concepts/metrics/index.md).

### Using Pre-Built Metrics

`ragas` comes with pre-built metrics for common evaluation tasks. For example, [AspectCritic](../concepts/metrics/available_metrics/aspect_critic.md) evaluates any aspect of your output:

```python
from ragas.metrics.collections import AspectCritic
from ragas.llms import instructor_llm_factory

# Setup your evaluator LLM
evaluator_llm = instructor_llm_factory("openai", model="gpt-4o")

# Use a pre-built metric
metric = AspectCritic(
    name="summary_accuracy",
    definition="Verify if the summary is accurate and captures key information.",
    llm=evaluator_llm
)

# Score your application's output
score = await metric.ascore(
    user_input="Summarize this text: ...",
    response="The summary of the text is..."
)
print(f"Score: {score.value}")  # 1 = pass, 0 = fail
print(f"Reason: {score.reason}")
```

Pre-built metrics like this save you from defining evaluation logic from scratch. Explore [all available metrics](../concepts/metrics/available_metrics/index.md).

!!! info
    There are many other types of metrics that are available in `ragas` (with and without `reference`), and you may also create your own metrics if none of those fits your case. To explore this more checkout [more on metrics](../concepts/metrics/index.md).

### Evaluating on a Dataset

In your quickstart project, you'll see in the `load_dataset()` function, which creates test data with multiple samples:

```python
from ragas import Dataset

# Create a dataset with multiple test samples
dataset = Dataset(
    name="test_dataset",
    backend="local/csv",  # Can also use JSONL, Google Drive, or in-memory
    root_dir=".",
)

# Add samples to the dataset
data_samples = [
    {
        "user_input": "What is ragas?",
        "response": "Ragas is an evaluation framework...",
        "expected": "Ragas provides objective metrics..."
    },
    {
        "user_input": "How do metrics work?",
        "response": "Metrics score your application...",
        "expected": "Metrics evaluate performance..."
    },
]

for sample in data_samples:
    dataset.append(sample)

# Save to disk
dataset.save()
```

This gives you multiple test cases instead of evaluating one example at a time. Learn more about [datasets and experiments](../concepts/components/eval_dataset.md).

Your generated project includes sample data in the `evals/datasets/` folder - you can edit those files to add more test cases.

### Want help in improving your AI application using evals?

In the past 2 years, we have seen and helped improve many AI applications using evals.

We are compressing this knowledge into a product to replace vibe checks with eval loops so that you can focus on building great AI applications.

If you want help with improving and scaling up your AI application using evals.


🔗 Book a [slot](https://bit.ly/3EBYq4J) or drop us a line: [founders@explodinggradients.com](mailto:founders@explodinggradients.com).


![](../_static/ragas_app.gif)




## Up Next

- [Evaluate a simple RAG application](rag_eval.md)

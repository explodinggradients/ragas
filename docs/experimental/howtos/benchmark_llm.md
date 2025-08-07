# How to Evaluate a New LLM For Your Use Case

When a new LLM is released, you might want to determine if it outperforms your current model for your specific use case. This guide shows you how to run an accuracy comparison between two models using Ragas experimental framework.

## What you'll accomplish

By the end of this guide, you'll have:

- Set up a structured evaluation comparing two LLMs
- Evaluated model performance on a realistic business task
- Generated detailed results to inform your model selection decision

## Prerequisites

1. Install ragas_experimental[examples]
```bash
pip install ragas_experimental[examples]
```
2. Setup your OpenAI API key
```bash
export OPENAI_API_KEY="your_openai_api_key"
```

## The evaluation scenario

We'll use eligibility reasoning as our test case: given a customer profile, determine if they qualify for a discount and explain why. This task requires rule-chaining and explanation - skills that differentiate model capabilities.

*Note: You can adapt this approach to any use case that matters for your application.*

> **💡 Quick Start**: If you want to see the complete evaluation in action, you can jump straight to the [end-to-end command](#running-the-evaluation-end-to-end) that runs everything and generates comparison results automatically.
> 
> **📁 Full Code**: The complete source code for this example is available on [Github](https://github.com/explodinggradients/ragas/tree/main/experimental/ragas_examples/benchmark_llm)

## Set up your environment and API access

First, ensure you have your API credentials configured:

```bash
export OPENAI_API_KEY=your_actual_api_key
```

## Configure the models to compare

The evaluation compares two models defined in the [config.py](https://github.com/explodinggradients/ragas/tree/main/experimental/ragas_examples/benchmark_llm/config.py):

```python
# Model configuration for evaluationing
BASELINE_MODEL = "gpt-4.1-mini" # Your current model
CANDIDATE_MODEL = "o4-mini"     # The new model to evaluate
```

The baseline model represents your current choice, while the candidate is the new model you're considering. You can modify these in your own implementation.

## Test your prompt setup

Before running the full evaluation, verify your setup works:

```bash
python -m ragas_examples.benchmark_llm.prompt
```

This will test a sample customer profile with both models to ensure:

- API keys are working
- Models are accessible  
- JSON output format is correct

You should see structured JSON responses from both models showing eligibility decisions and discount calculations.

## Examine the evaluation dataset

The evaluation uses a pre-built dataset with test cases that includes:

- Simple cases with clear outcomes
- Edge cases at rule boundaries  
- Complex scenarios with ambiguous information

Each case specifies:

- `customer_profile`: The input data
- `expected_eligible`: Whether discount applies (True/False)
- `expected_discount`: Expected discount percentage
- `description`: Case complexity indicator

Example dataset structure:

```csv
customer_profile,expected_eligible,expected_discount,description
"Customer: Age 67, Income $45k, Premium 3yrs",true,25,"Senior + Premium member"
"Customer: Age 25, Income $25k, Student",true,15,"Student discount case"
"Customer: Age 30, Income $50k, New member",true,5,"New customer case"
```

To customize the dataset for your use case, create a `datasets/` directory and add your own CSV file. Refer to [Datasets - Core Concepts](../core_concepts/datasets.md) for more information.

It is better to sample real data from your application to create the dataset. If that is not available, you can generate synthetic data using an LLM. Since our use case is slightly complex, we recommend using a model like o3 which can generate more accurate data. Always make sure to review and verify the data you use. 

### Load dataset

```python
def load_dataset():
    """Load the dataset from CSV file."""
    import os
    # Get the directory where this file is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    dataset = Dataset.load(
        name="eligibility_benchmark",
        backend="local/csv",
        root_dir=current_dir
    )
    return dataset
```

The dataset loader finds your CSV file in the `datasets/` directory and loads it for evaluation. 

### Metrics function

It is generaly better to use a simple metric. You should use a metric relevant to your use case. More information on metrics can be found in [Metrics - Core Concepts](../core_concepts/metrics.md). The evaluation uses this accuracy metric to score each response:

```python
@discrete_metric(name="eligibility_accuracy", allowed_values=["correct", "incorrect"])
def eligibility_accuracy(prediction: str, expected_eligible, expected_discount):
    """Check if the eligibility and discount prediction is correct."""
    parsed = parse_eligibility_response(prediction)
    
    # Convert expected values to correct types (CSV loads as strings)
    expected_eligible_bool = str(expected_eligible).lower() == 'true'
    expected_discount_int = int(expected_discount)
    
    eligible_correct = parsed['eligible'] == expected_eligible_bool
    discount_correct = parsed['discount'] == expected_discount_int
    
    if eligible_correct and discount_correct:
        return MetricResult(
            value="correct", 
            reason=f"Correctly identified eligible={expected_eligible_bool}, discount={expected_discount_int}%"
        )
    else:
        return MetricResult(
            value="incorrect",
            reason=f"Expected eligible={expected_eligible_bool}, discount={expected_discount_int}%; Got eligible={parsed['eligible']}, discount={parsed['discount']}%"
        )
```

### Experiment structure

Each model evaluation follows this experiment pattern:

```python
@experiment()
async def benchmark_experiment(row):
    # Get model response
    response = run_prompt(row["customer_profile"], model=model_name)
    
    # Parse response
    parsed = parse_eligibility_response(response)
    
    # Score the response
    score = eligibility_accuracy.score(
        prediction=response,
        expected_eligible=row["expected_eligible"],
        expected_discount=row["expected_discount"]
    )
    
    return {
        **row,
        "model": model_name,
        "response": response,
        "predicted_eligible": parsed['eligible'],
        "predicted_discount": parsed['discount'],
        "score": score.value,
        "score_reason": score.reason
    }
```

## Running the evaluation end-to-end

To run the complete evaluation with default settings:

1. Setup your OpenAI API key
```bash
export OPENAI_API_KEY="your_openai_api_key"
```
2. Run the evaluation
```bash
python -m ragas_examples.benchmark_llm.evals
```

This will:

- Load the dataset
- Run both baseline and candidate models on each test case
- Evaluate responses using accuracy metrics
- Generate detailed comparison results
- Save individual experiment results to CSV files

You can then inspect the results by opening the `experiments/` directory to see detailed per-case results for each model.


## Interpret results and make your decision

### Analyze the accuracy metrics

The evaluation provides three key numbers:

- **Baseline accuracy**: How well your current model performs
- **Candidate accuracy**: How well the new model performs  
- **Performance difference**: The gap between them

### Review detailed case results

Examine the detailed case-by-case breakdown to understand:

- Which types of problems each model struggles with
- Whether failures occur on simple or complex cases
- Consistency of performance across different scenarios

### Consider additional factors

While accuracy is crucial, also evaluate:

- **Cost**: Token pricing differences between models
- **Latency**: Response time requirements for your application

*In production systems, these factors often influence the final decision as much as raw accuracy.*

### Make your selection decision

Choose the candidate model if:

- It significantly outperforms the baseline
- The performance gain justifies any cost/latency tradeoffs
- It handles your most critical use cases reliably

Stick with your baseline if:

- Performance differences are minimal
- The candidate model fails on cases critical to your business
- Cost or latency constraints favor the current model

## Adapting to your use case

To evaluate models for your specific application, you can use the [GitHub code](https://github.com/explodinggradients/ragas/tree/main/experimental/ragas_examples/benchmark_llm) as a template:

1. **Replace the prompt**: Modify the system prompt in `prompt.py` for your task
2. **Update the dataset**: Create test cases relevant to your domain 
3. **Adjust the metric**: Replace the `eligibility_accuracy` function with your own scoring logic
4. **Update the experiment function**: Modify the experiment to call your custom metric
5. **Configure models**: Update the model names in `config.py`

The Ragas experimental framework handles the orchestration, parallel execution, and result aggregation automatically for you, helping you evaluate and focus on your use case!


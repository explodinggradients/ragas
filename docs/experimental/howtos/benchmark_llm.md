# How to Evaluate a New LLM For Your Use Case

When a new LLM is released, you might want to determine if it outperforms your current model for your specific use case. This guide shows you how to run an accuracy comparison between two models using Ragas framework.

## What you'll accomplish

By the end of this guide, you'll have:

- Set up a structured evaluation comparing two LLMs
- Evaluated model performance on a realistic business task
- Generated detailed results to inform your model selection decision
- A reusable evaluation loop you can rerun whenever new models drop

## The evaluation scenario

We'll use discount calculation as our test case: given a customer profile, calculate the appropriate discount percentage and explain the reasoning. This task requires rule application and reasoning - skills that differentiate model capabilities.

*Note: You can adapt this approach to any use case that matters for your application.*

> **📁 Full Code**: The complete source code for this example is available on [Github](https://github.com/explodinggradients/ragas/tree/main/experimental/ragas_examples/benchmark_llm)

## Set up your environment and API access

First, ensure you have your API credentials configured:

```bash
export OPENAI_API_KEY=your_actual_api_key
```

## Test your setup

Verify your setup works:

```bash
python -m ragas_examples.benchmark_llm.prompt
```

This will test a sample customer profile to ensure your setup works. 

You should see structured JSON responses showing discount calculations and reasoning.

??? example "Example output"
    ```bash
    $ python -m ragas_examples.benchmark_llm.prompt
    ```
    Output:
    ```
    === System Prompt ===

    You are a discount calculation assistant. I will provide a customer profile and you must calculate their discount percentage and explain your reasoning.

    Discount rules:
    - Age 65+ OR student status: 15% discount
    - Annual income < $30,000: 20% discount  
    - Premium member for 2+ years: 10% discount
    - New customer (< 6 months): 5% discount

    Rules can stack up to a maximum of 35% discount.

    Respond in JSON format only:
    {
    "discount_percentage": number,
    "reason": "clear explanation of which rules apply and calculations",
    "applied_rules": ["list", "of", "applied", "rule", "names"]
    }


    === Customer Profile ===

        Customer Profile:
        - Name: Sarah Johnson
        - Age: 67
        - Student: No
        - Annual Income: $45,000
        - Premium Member: Yes, for 3 years
        - Account Age: 3 years
        

    === Running Prompt with default model gpt-4.1-nano-2025-04-14 ===
    {
    "discount_percentage": 25,
    "reason": "Sarah qualifies for a 15% discount due to age (67). She also gets a 10% discount for being a premium member for over 2 years. The total stacking of 15% and 10% discounts results in 25%. No other discounts apply based on income or account age.",
    "applied_rules": ["Age 65+", "Premium member for 2+ years"]
    }
    ```

## Examine the evaluation dataset

For this evaluation we've built a synthetic dataset with test cases that includes:

- Simple cases with clear outcomes
- Edge cases at rule boundaries  
- Complex scenarios with ambiguous information

Each case specifies:

- `customer_profile`: The input data
- `expected_discount`: Expected discount percentage
- `description`: Case complexity indicator

Example dataset structure (add an `id` column for easy comparison):

| ID | Customer Profile | Expected Discount | Description |
|----|------------------|-------------------|-------------|
| 1 | Martha is a 70-year-old retiree who enjoys gardening. She has never enrolled in any academic course recently, has an annual pension of 50,000 dollars, signed up for our service nine years ago and never upgraded to premium. | 15 | Senior only |
| 2 | Arjun, aged 19, is a full-time computer-science undergraduate. His part-time job brings in about 45,000 dollars per year. He opened his account a year ago and has no premium membership. | 15 | Student only |
| 3 | Cynthia, a 40-year-old freelance artist, earns roughly 25,000 dollars a year. She is not studying anywhere, subscribed to our basic plan five years back and never upgraded to premium. | 20 | Low income only |

To customize the dataset for your use case, create a `datasets/` directory and add your own CSV file. Refer to [Datasets - Core Concepts](../core_concepts/datasets.md) for more information.

It is better to sample real data from your application to create the dataset. If that is not available, you can generate synthetic data using an LLM. Since our use case is slightly complex, we recommend using a model like gpt-5-high which can generate more accurate data. Always make sure to manually review and verify the data you use. 

!!! note
    While the example dataset here has roughly 10 cases to keep the guide compact, you can start small with 20-30 samples for a real-world evaluation, but make sure you slowly iterate to improve it to the 50-100 samples range to get more trustable results from evaluation. Ensure broad coverage of the different scenarios your agent may face (including edge cases and complex questions). Your accuracy does not need to be 100% initially—use the results for error analysis, iterate on prompts, data, and tools, and keep improving.

### Load dataset

```python
def load_dataset():
    """Load the dataset from CSV file."""
    import os
    # Get the directory where this file is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    dataset = Dataset.load(
        name="discount_benchmark",
        backend="local/csv",
        root_dir=current_dir
    )
    return dataset
```

The dataset loader finds your CSV file in the `datasets/` directory and loads it for evaluation. 

### Metrics function

It is generally better to use a simple metric. You should use a metric relevant to your use case. More information on metrics can be found in [Metrics - Core Concepts](../core_concepts/metrics.md). The evaluation uses this accuracy metric to score each response:

```python
@discrete_metric(name="discount_accuracy", allowed_values=["correct", "incorrect"])
def discount_accuracy(prediction: str, expected_discount):
    """Check if the discount prediction is correct."""
    import json
    
    parsed_json = json.loads(prediction)
    predicted_discount = parsed_json.get("discount_percentage")
    expected_discount_int = int(expected_discount)
    
    if predicted_discount == expected_discount_int:
        return MetricResult(
            value="correct", 
            reason=f"Correctly calculated discount={expected_discount_int}%"
        )
    else:
        return MetricResult(
            value="incorrect",
            reason=f"Expected discount={expected_discount_int}%; Got discount={predicted_discount}%"
        )
```

### Experiment structure

Each model evaluation follows this experiment pattern:

```python
@experiment()
async def benchmark_experiment(row):
    # Get model response (run in thread to keep async runner responsive)
    response = await asyncio.to_thread(run_prompt, row["customer_profile"], model=model_name)
    
    # Parse response (strict JSON mode expected)
    try:
        parsed_json = json.loads(response)
        predicted_discount = parsed_json.get('discount_percentage')
    except Exception:
        predicted_discount = None
    
    # Score the response
    score = discount_accuracy.score(
        prediction=response,
        expected_discount=row["expected_discount"]
    )
    
    return {
        **row,
        "model": model_name,
        "experiment_name": experiment_name,
        "response": response,
        "predicted_discount": predicted_discount,
        "score": score.value,
        "score_reason": score.reason
    }

return benchmark_experiment
```

## Run experiments

We've setup the main evals code such that you can run it with different models using CLI. We’ll use these example models:

- Baseline: "gpt-4.1-nano-2025-04-14"
- Candidate: "gpt-5-nano-2025-08-07"

```bash
# Baseline
python -m ragas_examples.benchmark_llm.evals run --model "gpt-4.1-nano-2025-04-14"

# Candidate
python -m ragas_examples.benchmark_llm.evals run --model "gpt-5-nano-2025-08-07"
```

Each command saves a CSV under `experiments/` with per-row results, including:

- id, model, experiment_name, response, predicted_discount, score, score_reason

??? example "Sample experiment output (only showing few columns for readability)"
    | ID | Description | Expected | Predicted | Score | Score Reason |
    |----|-------------|----------|-----------|-------|--------------|
    | 1 | Senior only | 15 | 15 | correct | Correctly calculated discount=15% |
    | 2 | Student only | 15 | 5 | incorrect | Expected discount=15%; Got discount=5% |
    | 3 | Low income only | 20 | 20 | correct | Correctly calculated discount=20% |
    | 4 | Senior, low income, new customer (capped) | 35 | 35 | correct | Correctly calculated discount=35% |
    | 6 | Premium 2+ yrs only | 10 | 15 | incorrect | Expected discount=10%; Got discount=15% |


!!! note
    When possible, pin and record the exact model snapshot/version (for example, "gpt-4o-2024-08-06" instead of just "gpt-4o"). Providers regularly update alias names, and performance can change between snapshots. You can find available snapshots in the provider's model documentation (see OpenAI's [model catalog](https://platform.openai.com/docs/models) as an example). Including the snapshot in your results makes future comparisons fair and reproducible.


## Compare results

After running experiments with different models, you'll want to see how they performed side-by-side. We've setup a simple compare command that takes your experiment results and puts them in one file so you can easily see which model did better on each test case. You can open this in your CSV viewer (Excel/Google Sheets/Numbers) to review.

```bash
python -m ragas_examples.benchmark_llm.evals compare \
  --inputs 'experiments/exp1.csv' 'experiments/exp2.csv'
```

This command will:

- Read both experiment files
- Print the accuracy for each model
- Create a new comparison file with results side-by-side

The comparison file shows:

- The test case details (customer profile, expected discount)
- For each model: its response, whether it was correct, and why

??? example "Example evaluation output"
    ```bash
    $ python -m ragas_examples.benchmark_llm.evals compare \
      --inputs 'experiments/20250820-145315-gpt-4.1-nano-2025-04-14.csv' 'experiments/20250820-145327-gpt-5-nano-2025-08-07.csv'
    ```
    
    ```
    gpt-4.1-nano-2025-04-14 Accuracy: 50.00%
    gpt-5-nano-2025-08-07 Accuracy: 90.00%
    Combined comparison saved to: experiments/20250820-150548-comparison.csv
    ```

### Analyze results with the combined CSV

In this example run:

- Filtering for cases where one model outperforms the other surfaces these cases: "Senior and new customer", "Student and new customer", "Student only", "Premium 2+ yrs only".
- The reason field in each model's response shows why it gave the output it did. 

??? example "Sample rows from comparison CSV (showing limited columns for readability)"
    | id | customer_profile | description | expected_discount | gpt-4.1-nano-2025-04-14_score | gpt-5-nano-2025-08-07_score | gpt-4.1-nano-2025-04-14_score_reason | gpt-5-nano-2025-08-07_score_reason | gpt-4.1-nano-2025-04-14_response | gpt-5-nano-2025-08-07_response |
    |---:|---|---|---:|---|---|---|---|---|---|
    | 2 | Arjun, aged 19, is a full-time computer-science undergraduate. His part-time job brings in about 45,000 dollars per year. He opened his account a year ago and has no premium membership. | Student only | 15 | incorrect | correct | Expected discount=15%; Got discount=0% | Correctly calculated discount=15% | ...reason="Arjun is 19 years old, so he does not qualify for age-based or senior discounts. His annual income of $45,000 exceeds the $30,000 threshold, so no income-based discount applies. He opened his account a year ago, which is more than 6 months, so he is not a new customer. He has no premium membership and no other applicable discounts."... | ...reason="Eligible for 15% discount due to student status (Arjun is 19 and an undergraduate)."... |
    | 6 | Leonardo is 64, turning 65 next month. His salary is exactly 30,000 dollars. He has maintained a premium subscription for two years and seven months and has been with us for five years. | Premium 2+ yrs only | 10 | incorrect | correct | Expected discount=10%; Got discount=25% | Correctly calculated discount=10% | ...reason="Leonardo is about to turn 65, so he qualifies for the age discount of 15%. Premium 2+ years noted"... | ...reason="Leonardo is 64, turning 65 next month. premium 2+ years: 10%"... |

!!! tip "Re-run when new models drop"
    Once this evaluation lives alongside your project, it becomes a repeatable check. When a new LLM is released (often weekly nowadays), plug it in as the candidate and rerun the same evaluation to compare against your current baseline.


## Interpret results and make your decision

### What to look at
- **Baseline accuracy** vs **Candidate accuracy** and the **difference**.
  - Example from this run: baseline 50% (5/10), candidate 90% (9/10), difference +40%.

### How to read the rows
- Skim rows where the two models disagree.
- Use each row’s score_reason to see why it was marked correct/incorrect.
- Look for patterns (e.g., missed rule stacking, boundary cases like “almost 65”, exact income thresholds).

### Beyond accuracy
- Check **cost** and **latency**. Higher accuracy may not be worth it if it’s too slow or too expensive for your use case.

### Decide
- Switch if the new model is clearly more accurate on your important cases and fits your cost/latency needs.
- Stay if gains are small, failures hit critical cases, or cost/latency are not acceptable.

In this example:
- We would switch to "gpt-5-nano-2025-08-07". It improves accuracy from 50% to 90% (+40%) and fixes the key failure modes (missed rule stacking, boundary conditions). If its latency/cost fits your constraints, it’s the better default.

## Adapting to your use case

To evaluate models for your specific application, you can use the [GitHub code](https://github.com/explodinggradients/ragas/tree/main/experimental/ragas_examples/benchmark_llm) as a template and adapt it to your use case.

The Ragas framework handles the orchestration, parallel execution, and result aggregation automatically for you, helping you evaluate and focus on your use case!


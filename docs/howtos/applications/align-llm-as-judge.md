# How to Align an LLM as a Judge

In this guide, you'll learn how to systematically evaluate and align an LLM-as-judge metric with human expert judgments using Ragas.

- Build a reusable evaluation pipeline for judge alignment
- Analyze disagreement patterns between judge and human labels
- Iterate on judge prompts to improve alignment with expert decisions

## Why align your LLM judge first?

Before running evaluation experiments, align your LLM judge to your specific use case. A misaligned judge is like a compass pointing the wrong way - every improvement you make based on its guidance moves you further from your goal. Aligning the judge to match expert judgments ensures you're improving what actually matters. This alignment step is the foundation of reliable evaluation. 

## Setup your environment

We've created a simple module you can install and run so that you can focus on understanding the evaluation process instead of creating the application.

```bash
uv pip install "ragas[examples]"
export OPENAI_API_KEY="your-api-key-here"
```

!!! note "Full code"
    You can view the full code for the judge alignment evaluation pipeline [here](https://github.com/explodinggradients/ragas/tree/main/examples/ragas_examples/judge_alignment).

## Understand the dataset

We'll use the [EvalsBench dataset](https://github.com/explodinggradients/EvalsBench/blob/main/data/benchmark_df.csv) which contains expert-annotated examples of LLM responses to business questions. Each row includes:

- `question`: The original question asked
- `grading_notes`: Key points that should be covered in a good response
- `response`: The LLM's generated response
- `target`: Human expert's binary judgment (pass/fail)

**Download the dataset:**

```bash
# Create datasets folder and download the dataset
mkdir -p datasets
curl -o datasets/benchmark_df.csv https://raw.githubusercontent.com/explodinggradients/EvalsBench/main/data/benchmark_df.csv
```

**Load and examine the dataset:**

```python
import pandas as pd
from ragas import Dataset

def load_dataset(csv_path: str = None) -> Dataset:
    """Load annotated dataset with human judgments.
    
    Expected columns: question, grading_notes, response, target (pass/fail)
    """
    path = csv_path or "datasets/benchmark_df.csv"
    df = pd.read_csv(path)

    dataset = Dataset(name="llm_judge_alignment", backend="local/csv")
    
    for _, row in df.iterrows():
        dataset.append({
            "question": row["question"],
            "grading_notes": row["grading_notes"],
            "response": row["response"],
            "target": (row["target"]),
        })
    
    return dataset

# Load the dataset
dataset = load_dataset()
print(f"Dataset loaded with {len(dataset)} samples")
```

**Sample rows from the dataset:**

| question | grading_notes | response | target |
|----------|---------------|----------|---------|
| What are the key methods for determining the pre-money valuation of a tech startup before a Series A investment round, and how do they differ? | DCF method: !future cash flows!, requires projections; Comp. analysis: similar co. multiples; VC method: rev x multiple - post-$; *Founder's share matter*; strategic buyers pay more. | Determining the pre-money valuation of a tech startup before a Series A investment round is a critical step... (covers DCF, comparable analysis, VC method) | pass |
| What key metrics and strategies should a startup prioritize to effectively manage and reduce churn rate in a subscription-based business model? | Churn:! monitor monthly, <5% ideal. *Retention strategies*: engage users, improve onboarding. CAC & LTV: balance 3:1+. Feedback loops: implement early. *Customer support*: proactive & responsive, critical. | Managing and reducing churn rate in a subscription-based business model is crucial... (missing specific metrics and strategies) | fail |

The dataset includes multiple responses to the same questions - some pass and others fail. This helps the judge learn nuanced distinctions between acceptable and unacceptable responses.

## Understand the evaluation approach

In this guide, we evaluate pre-existing responses from the dataset rather than generating new ones. This approach ensures reproducible results across evaluation runs, allows us to focus on judge alignment rather than response generation.

The evaluation workflow is: **Dataset row (question + response) → Judge → Compare with human target**

## Define evaluation metrics

For judge alignment, we need two metrics:

**Primary metric: `accuracy` (LLM judge)** - Evaluates responses and returns pass/fail decisions with critiques.

**Alignment metric: `judge_alignment`** - Checks if the judge's decision matches the human expert's verdict.

### Setting up the judge metric

Define a simple baseline judge metric that evaluates responses against grading notes:

```python
from ragas.metrics import DiscreteMetric

# Define the judge metric with a simple baseline prompt
accuracy_metric = DiscreteMetric(
    name="accuracy",
    prompt="Check if the response contains points mentioned from the grading notes and return 'pass' or 'fail'.\n\nResponse: {response}\nGrading Notes: {grading_notes}",
    allowed_values=["pass", "fail"],
)
```

### The alignment metric

The alignment metric compares the judge's decision with the human verdict:

```python
from ragas.metrics.discrete import discrete_metric
from ragas.metrics.result import MetricResult

@discrete_metric(name="judge_alignment", allowed_values=["pass", "fail"])
def judge_alignment(judge_label: str, human_label: str) -> MetricResult:
    """Compare judge decision with human label."""
    judge = judge_label.strip().lower()
    human = human_label.strip().lower()
    
    if judge == human:
        return MetricResult(value="pass", reason=f"Judge={judge}; Human={human}")
    
    return MetricResult(value="fail", reason=f"Judge={judge}; Human={human}")
```

## The experiment function

The [experiment function](/concepts/experimentation) orchestrates the complete evaluation pipeline - evaluating responses with the judge and measuring alignment:

```python
from typing import Dict, Any
from ragas import experiment
from ragas.metrics import DiscreteMetric
from ragas_examples.judge_alignment import judge_alignment

@experiment()
async def judge_experiment(
    row: Dict[str, Any],
    accuracy_metric: DiscreteMetric,
    llm,
):
    """Run complete evaluation: Judge → Compare with human."""
    # Step 1: Get response (in production, this is where you'd call your LLM app)
    # For this evaluation, we use pre-existing responses from the dataset
    app_response = row["response"]
    
    # Step 2: Judge evaluates the response
    judge_score = await accuracy_metric.ascore(
        question=row["question"],
        grading_notes=row["grading_notes"],
        response=app_response,
        llm=llm,
    )

    # Step 3: Compare judge decision with human target
    alignment = judge_alignment.score(
        judge_label=judge_score.value,
        human_label=row["target"]
    )

    return {
        **row,
        "judge_label": judge_score.value,
        "judge_critique": judge_score.reason,
        "alignment": alignment.value,
        "alignment_reason": alignment.reason,
    }
```

## Run baseline evaluation

### Execute evaluation pipeline and collect results

```python
import os
from openai import AsyncOpenAI
from ragas.llms import instructor_llm_factory
from ragas_examples.judge_alignment import load_dataset

# Load dataset
dataset = load_dataset()
print(f"Dataset loaded with {len(dataset)} samples")

# Initialize LLM client
openai_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
llm = instructor_llm_factory("openai", model="gpt-5-mini", client=openai_client)

# Run the experiment
results = await judge_experiment.arun(
    dataset,
    name="judge_baseline_v1",
    accuracy_metric=accuracy_metric,
    llm=llm,
)

# Calculate alignment rate
passed = sum(1 for r in results if r["alignment"] == "pass")
total = len(results)
print(f"✅ Baseline alignment: {passed}/{total} passed ({passed/total:.1%})")
```


### Initial performance analysis


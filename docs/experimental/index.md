# Ragas Experimental

# ✨ Introduction


<div class="grid cards" markdown>
- 🚀 **Tutorials**

    Install with `pip` and get started with Ragas with these tutorials.

    [:octicons-arrow-right-24: Tutorials](tutorials/index.md)

- 📚 **Core Concepts**

    In depth explanation and discussion of the concepts and working of different features available in Ragas.

    [:octicons-arrow-right-24: Core Concepts](core_concepts/index.md)


</div>

## Installation

- Install ragas_experimental from pip

```bash
pip install ragas_experimental
```

- Install from source

```bash
git clone https://github.com/explodinggradients/ragas
```

```bash
cd ragas/experimental && pip install -e .
```


## Hello World 👋

Copy this snippet to a file named `hello_world.py` and run `python hello_world.py` 

```python
import numpy as np
from ragas_experimental import experiment, Dataset
from ragas_experimental.metrics import MetricResult, numeric_metric  


@numeric_metric(name="accuracy_score", allowed_values=(0, 1))
def accuracy_score(response: str, expected: str):
    result = 1 if expected.lower().strip() == response.lower().strip() else 0
    return MetricResult(result=result, reason=f"Match: {result == 1}")

def mock_app_endpoint(**kwargs) -> str:
    return np.random.choice(["Paris", "4", "Blue Whale", "Einstein", "Python"])

@experiment()
async def run_experiment(row):
    response = mock_app_endpoint(query=row.get("query"))
    accuracy = accuracy_score.score(response=response, expected=row.get("expected_output"))
    return {**row, "response": response, "accuracy": accuracy.value}

if __name__ == "__main__":
    import asyncio
    
    # Create dataset inline
    dataset = Dataset(name="test_dataset", backend="local/csv", root_dir=".")
    test_data = [
        {"query": "What is the capital of France?", "expected_output": "Paris"},
        {"query": "What is 2 + 2?", "expected_output": "4"},
        {"query": "What is the largest animal?", "expected_output": "Blue Whale"},
        {"query": "Who developed the theory of relativity?", "expected_output": "Einstein"},
        {"query": "What programming language is named after a snake?", "expected_output": "Python"},
    ]
    
    for sample in test_data:
        dataset.append(sample)
    dataset.save()
    
    # Run experiment
    results = asyncio.run(run_experiment.arun(dataset, name="first_experiment"))
```

View Results 

```
├── datasets
│   └── test_dataset.csv
└── experiments
    └── first_experiment.csv
```

Open the results in a CSV file

```bash
open experiments/first_experiment.csv
```
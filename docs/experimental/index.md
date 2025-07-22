# Ragas Experimental

# ✨ Introduction


<div class="grid cards" markdown>
- 🚀 **Tutorials**

    Install with `pip` and get started with Ragas with these tutorials.

    [:octicons-arrow-right-24: Tutorials](tutorials/index.md)

- 📚 **Core Concepts**

    In depth explanation and discussion of the concepts and working of different features available in Ragas.

    [:octicons-arrow-right-24: Core Concepts](core_concepts/index.md)




## Hello World 👋

![](hello_world.gif)

1. Install Ragas Experimental with local backend

```bash
pip install ragas-experimental && pip install "ragas-experimental[local]"
```


3. Create a simple experiment with a mock application endpoint, a dataset and a custom metric for accuracy.

Copy this snippet to a file named `hello_world.py` and run `python hello_world.py` 

```python
import numpy as np
from ragas_experimental import experiment, Dataset
from ragas_experimental.metrics import MetricResult, discrete_metric  


@discrete_metric(name="accuracy_score", allowed_values=["pass", "fail"])
def accuracy_score(response: str, expected: str):
    result = "pass" if expected.lower().strip() == response.lower().strip() else "fail"
    return MetricResult(value=result, reason=f"Match: {result == 'pass'}")

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

4. Run the script

```bash
python hello_world.py
```

5. Check the results

```bash
tree .
```

```
├── datasets
│   └── test_dataset.csv
└── experiments
    └── first_experiment.csv
```

6. View the results of your first experiment

```bash
open experiments/first_experiment.csv
```
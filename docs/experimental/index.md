# Ragas Experimental




## Hello World 
- Copy the following code into `evals.py`:

```py
import typing as t

import numpy as np
from ragas_experimental import BaseModel, Project
from ragas_experimental.metric import MetricResult, numeric_metric

p = Project(
    project_id="hello_world",
    backend="local",
    root_dir=".",
)


@numeric_metric(name="accuracy_score", range=(0, 1))
def accuracy_score(response: str, expected: str):
    """
    Is the response a good response to the query?
    """
    result = 1 if expected in response else 0
    return MetricResult(
        result=result,
        reason=(
            f"Response contains {expected}"
            if result
            else f"Response does not contain {expected}"
        ),
    )


def mock_app_endpoint(**kwargs) -> str:
    """Mock AI endpoint for testing purposes."""
    mock_responses = [
        "Paris","4","Blue Whale","Einstein","Python","Mount Everest","Shakespeare",
        "Mars","Apple","Leonardo da Vinci",]
    return np.random.choice(mock_responses)


class TestDataRow(BaseModel):
    id: t.Optional[int]
    query: str
    expected_output: str


class ExperimentDataRow(TestDataRow):
    response: str
    accuracy: int
    accuracy_reason: t.Optional[str] = None


@p.experiment(ExperimentDataRow)
async def run_experiment(row: TestDataRow):
    response = mock_app_endpoint(query=row.query)
    accuracy = accuracy_score.score(response=response, expected=row.expected_output)

    experiment_view = ExperimentDataRow(
        **row.model_dump(),
        response=response,
        accuracy=accuracy.result,
        accuracy_reason=accuracy.reason,
    )
    return experiment_view
```

- Run your first experiment:

```sh
ragas evals evals.py --dataset hello_world  --metrics accuracy
```

```bash
Running evaluation: evals.py
Dataset: hello_world
Getting dataset: hello_world
✓ Loaded dataset with 10 rows
✓ Completed experiments successfully
╭────────────────────────── Ragas Evaluation Results ──────────────────────────╮
│ Experiment: mystifying_diffie                                                │
│ Dataset: hello_world (10 rows)                                               │
╰──────────────────────────────────────────────────────────────────────────────╯
  Numerical Metrics   
┏━━━━━━━━━━┳━━━━━━━━━┓
┃ Metric   ┃ Current ┃
┡━━━━━━━━━━╇━━━━━━━━━┩
│ accuracy │   0.000 │
└──────────┴─────────┘
✓ Experiment results displayed
✓ Evaluation completed successfully
```

- View the results by opening the `hello_world/experiments` directory:

```sh
❯ tree hello_world
hello_world
├── datasets
│   └── hello_world.csv
├── experiments
│   ├── mystifying_diffie.csv
```

![](./experimental/hello_world.png)

- Run your next experiment and compare the results:

```sh
ragas evals evals.py --dataset hello_world --metrics accuracy --baseline latest
```

```bash
Running evaluation: evals.py
Dataset: hello_world
Baseline: mystifying_diffie
Getting dataset: hello_world
✓ Loaded dataset with 10 rows
✓ Completed experiments successfully
Comparing against baseline: mystifying_diffie
╭────────────────────────── Ragas Evaluation Results ──────────────────────────╮
│ Experiment: competent_cerf                                                   │
│ Dataset: hello_world (10 rows)                                               │
│ Baseline: mystifying_diffie                                                     │
╰──────────────────────────────────────────────────────────────────────────────╯
                Numerical Metrics                
┏━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━┳━━━━━━┓
┃ Metric   ┃ Current ┃ Baseline ┃  Delta ┃ Gate ┃
┡━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━╇━━━━━━┩
│ accuracy │   0.100 │    0.000 │ ▲0.100 │ pass │
└──────────┴─────────┴──────────┴────────┴──────┘
✓ Comparison completed
✓ Evaluation completed successfully
```
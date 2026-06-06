---
search:
  exclude: true
---

# Adding to your CI pipeline with Pytest

You can add Ragas evaluations as part of your Continuous Integration pipeline
to keep track of the qualitative performance of your RAG pipeline. Consider these as
part of your end-to-end test suite which you run before major changes and releases.

The most direct way to fail a CI job on quality regressions is to assert thresholds
against the aggregate metric scores in an `EvaluationResult`.

You can easily write a Pytest test as follows

!!! note
    This dataset that is already populated with outputs from a reference RAG
    When testing your own system make sure you use outputs from RAG pipeline
    you want to test. For more information on how to build your datasets check
    [Building HF `Dataset` with your own Data](./data_preparation.md) docs.

```python
from datasets import load_dataset

from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)


def test_amnesty_e2e():
    # loading the V2 dataset
    amnesty_qa = load_dataset("vibrantlabsai/amnesty_qa", "english_v2")["eval"]


    result = evaluate(
        amnesty_qa,
        metrics=[answer_relevancy, faithfulness, context_recall, context_precision],
    )
    result.assert_thresholds(
        min_scores={
            "answer_relevancy": 0.9,
            "context_recall": 0.95,
            "context_precision": 0.95,
        },
        ranges={
            "faithfulness": (0.3, 0.5),
        },
    )
```

You can also inspect a threshold report without raising immediately:

```python
report = result.check_thresholds(
    min_scores={"faithfulness": 0.8}
)

assert report["passed"], report["failures"]
```

`result.summary()` returns aggregate metric values as a plain dictionary.
`result["metric_name"]` still returns the per-row scores for that metric, and
`result.to_pandas()` remains useful when you want a dataframe with the input rows and
their metric scores.

## Using Pytest Markers for Ragas E2E tests

Because these are long end-to-end test one thing that you can leverage is [Pytest Markers](https://docs.pytest.org/en/latest/example/markers.html) which help you mark your tests with special tags. It is recommended to mark Ragas tests with special tags, so you can run them only when needed.

To add a new `ragas_ci` tag to Pytest, add the following to your `conftest.py`
```python
def pytest_configure(config):
    """
    configure pytest
    """
    # add `ragas_ci`
    config.addinivalue_line(
        "markers", "ragas_ci: Set of tests that will be run as part of Ragas CI"
    )
```

now you can use `ragas_ci` to mark all the tests that are part of Ragas CI.

```python
import pytest
from datasets import load_dataset

from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)


@pytest.mark.ragas_ci
def test_amnesty_e2e():
    # loading the V2 dataset
    amnesty_qa = load_dataset("vibrantlabsai/amnesty_qa", "english_v2")["eval"]


    result = evaluate(
        amnesty_qa,
        metrics=[answer_relevancy, faithfulness, context_recall, context_precision],
    )
    result.assert_thresholds(
        min_scores={
            "answer_relevancy": 0.9,
            "context_recall": 0.95,
            "context_precision": 0.95,
        },
        ranges={
            "faithfulness": (0.3, 0.5),
        },
    )
```

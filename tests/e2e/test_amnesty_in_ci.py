import pytest
from datasets import load_dataset

from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)

# loading the V2 dataset
amnesty_qa = load_dataset("explodinggradients/amnesty_qa", "english_v2")["eval"]


def assert_in_range(score: float, value: float, max_range: float):
    """
    Check if computed score is within the range of value +/- max_range
    """
    assert value - max_range <= score <= value + max_range


@pytest.mark.ragas_ci
def test_amnesty_e2e():
    result = evaluate(
        amnesty_qa,
        metrics=[answer_relevancy, faithfulness, context_recall, context_precision],
        in_ci=True,
    )
    assert result["answer_relevancy"] >= 0.9
    assert result["context_recall"] >= 0.95
    assert result["context_precision"] >= 0.95
    assert_in_range(result["faithfulness"], value=0.4, max_range=0.1)

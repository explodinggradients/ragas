import typing as t

import pytest
from datasets import load_dataset

from ragas import EvaluationDataset, evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

if t.TYPE_CHECKING:
    from datasets import Dataset

# loading the V2 dataset
amnesty_qa = load_dataset("explodinggradients/amnesty_qa", "english_v3")["eval"]  # type: ignore


def assert_in_range(score: float, value: float, plus_or_minus: float):
    """
    Check if computed score is within the range of value +/- max_range
    """
    assert value - plus_or_minus <= score <= value + plus_or_minus


@pytest.mark.ragas_ci
def test_amnesty_e2e():
    result = evaluate(
        EvaluationDataset.from_hf_dataset(t.cast("Dataset", amnesty_qa))[:1],
        metrics=[answer_relevancy, faithfulness, context_recall, context_precision],
        in_ci=True,
        show_progress=False,
    )
    assert result is not None


@pytest.mark.ragas_ci
def test_assert_in_range():
    assert_in_range(0.51, value=0.5, plus_or_minus=0.1)

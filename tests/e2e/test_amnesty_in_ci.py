import os
import typing as t

import pytest

from ragas import EvaluationDataset, evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from tests.e2e.test_dataset_utils import load_amnesty_dataset_safe

if t.TYPE_CHECKING:
    from datasets import Dataset

# loading the dataset
amnesty_qa = load_amnesty_dataset_safe("english_v3")  # type: ignore


def assert_in_range(score: float, value: float, plus_or_minus: float):
    """
    Check if computed score is within the range of value +/- max_range
    """
    assert value - plus_or_minus <= score <= value + plus_or_minus


@pytest.mark.ragas_ci
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_amnesty_e2e():
    result = evaluate(
        EvaluationDataset.from_hf_dataset(t.cast("Dataset", amnesty_qa))[:1],
        metrics=[answer_relevancy, faithfulness, context_recall, context_precision],
        show_progress=False,
    )
    assert result is not None


@pytest.mark.ragas_ci
def test_assert_in_range():
    assert_in_range(0.51, value=0.5, plus_or_minus=0.1)

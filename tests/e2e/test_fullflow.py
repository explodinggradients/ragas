import os
import typing as t

import pytest

from ragas import EvaluationDataset, evaluate
from ragas.metrics import answer_relevancy, context_precision, faithfulness
from ragas.metrics._aspect_critic import harmfulness
from tests.e2e.test_dataset_utils import load_amnesty_dataset_safe

if t.TYPE_CHECKING:
    from datasets import Dataset


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_evaluate_e2e():
    ds = load_amnesty_dataset_safe("english_v3")  # type: ignore
    result = evaluate(
        EvaluationDataset.from_hf_dataset(t.cast("Dataset", ds))[:1],
        metrics=[answer_relevancy, context_precision, faithfulness, harmfulness],
        show_progress=False,
    )
    assert result is not None

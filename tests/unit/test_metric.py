from ragas.metrics.base import EvaluationMode
from ragas.metrics.utils import get_available_metrics


def test_get_available_metrics():
    from datasets import Dataset

    ds = Dataset.from_dict({"question": ["a", "b", "c"], "answer": ["d", "e", "f"]})
    assert all(
        [
            metric.evaluation_mode == EvaluationMode.qa
            for metric in get_available_metrics(ds)
        ]
    ), "All metrics should have evaluation mode qa"

    ds = Dataset.from_dict(
        {
            "question": ["a", "b", "c"],
            "answer": ["d", "e", "f"],
            "contexts": ["g", "h", "i"],
        }
    )
    assert all(
        [
            metric.evaluation_mode
            in [EvaluationMode.qa, EvaluationMode.qc, EvaluationMode.qac]
            for metric in get_available_metrics(ds)
        ]
    ), "All metrics should have evaluation mode qa"

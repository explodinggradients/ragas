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


def test_metric():
    from ragas.metrics.base import Metric

    class FakeMetric(Metric):
        name = "fake_metric"  # type: ignore
        evaluation_mode = EvaluationMode.qa  # type: ignore

        def init(self, run_config):
            pass

        async def _ascore(self, row, callbacks) -> float:
            return 0

    fm = FakeMetric()
    assert fm.score({"question": "a", "answer": "b"}) == 0

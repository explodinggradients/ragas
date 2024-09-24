import typing as t
from dataclasses import dataclass, field

from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.metrics.base import MetricType
from ragas.metrics.utils import get_available_metrics


def test_get_available_metrics():
    sample1 = SingleTurnSample(user_input="What is X", response="Y")
    sample2 = SingleTurnSample(user_input="What is Z", response="W")
    ds = EvaluationDataset(samples=[sample1, sample2])

    assert all(
        [
            m.required_columns["SINGLE_TURN"] == {"response", "user_input"}
            for m in get_available_metrics(ds)
        ]
    ), "All metrics should have required columns ('user_input', 'response')"


def test_metric():
    from ragas.metrics.base import Metric

    @dataclass
    class FakeMetric(Metric):
        name = "fake_metric"  # type: ignore
        _required_columns: t.Dict[MetricType, t.Set[str]] = field(
            default_factory=lambda: {MetricType.SINGLE_TURN: {"user_input", "response"}}
        )

        def init(self, run_config):
            pass

        async def _ascore(self, row, callbacks) -> float:
            return 0

    fm = FakeMetric()
    assert fm.score({"user_input": "a", "response": "b"}) == 0


def test_single_turn_metric():
    from ragas.metrics.base import SingleTurnMetric

    class FakeMetric(SingleTurnMetric):
        name = "fake_metric"
        _required_columns = {MetricType.SINGLE_TURN: {"user_input", "response"}}

        def init(self, run_config):
            pass

        async def _ascore(self, row, callbacks) -> float:
            pass

        async def _single_turn_ascore(self, sample: SingleTurnSample, callbacks):
            return 0

    fm = FakeMetric()
    assert (
        fm.single_turn_score(SingleTurnSample(**{"user_input": "a", "response": "b"}))
        == 0
    )

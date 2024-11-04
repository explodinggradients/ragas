import typing as t
from dataclasses import dataclass, field

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics.base import MetricType


def test_single_turn_metric():
    from ragas.metrics.base import SingleTurnMetric

    class FakeMetric(SingleTurnMetric):
        name = "fake_metric"  # type: ignore
        _required_columns = {MetricType.SINGLE_TURN: {"user_input", "response"}}

        def init(self, run_config):
            pass

        async def _ascore(self, row, callbacks) -> float:
            return 0

        async def _single_turn_ascore(self, sample: SingleTurnSample, callbacks):
            return 0

    fm = FakeMetric()
    assert fm.single_turn_score(SingleTurnSample(user_input="a", response="b")) == 0


def test_required_columns():
    from ragas.metrics.base import MetricType, SingleTurnMetric

    @dataclass
    class FakeMetric(SingleTurnMetric):
        name = "fake_metric"  # type: ignore
        _required_columns: t.Dict[MetricType, t.Set[str]] = field(
            default_factory=lambda: {
                MetricType.SINGLE_TURN: {
                    "user_input",
                    "response",
                    "retrieved_contexts:optional",
                },
            }
        )

        def init(self, run_config):
            pass

        async def _ascore(self, row, callbacks) -> float:
            return 0

        async def _single_turn_ascore(self, sample: SingleTurnSample, callbacks):
            return 0

    fm = FakeMetric()

    # only return required columns, don't include optional columns
    assert fm.required_columns[MetricType.SINGLE_TURN.name] == {
        "user_input",
        "response",
    }

    # check if optional columns are included
    assert fm.get_required_columns(with_optional=False)[
        MetricType.SINGLE_TURN.name
    ] == {
        "user_input",
        "response",
    }
    # check if optional columns are included
    assert fm.get_required_columns(with_optional=True)[MetricType.SINGLE_TURN.name] == {
        "user_input",
        "response",
        "retrieved_contexts",
    }

    # check if only required columns are returned
    assert (
        fm._only_required_columns_single_turn(
            SingleTurnSample(user_input="a", response="b", reference="c")
        ).to_dict()
        == SingleTurnSample(user_input="a", response="b").to_dict()
    )

    # check if optional columns are included if they are not none
    assert (
        fm._only_required_columns_single_turn(
            SingleTurnSample(user_input="a", response="b", retrieved_contexts=["c"])
        ).to_dict()
        == SingleTurnSample(
            user_input="a", response="b", retrieved_contexts=["c"]
        ).to_dict()
    )

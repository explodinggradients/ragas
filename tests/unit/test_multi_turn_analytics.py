from __future__ import annotations

import typing as t

import pytest

from ragas._analytics import EvaluationEvent
from ragas.dataset_schema import MultiTurnSample
from ragas.messages import AIMessage
from ragas.metrics.base import MetricType, MultiTurnMetric
from ragas.run_config import RunConfig


class _StubMultiTurnMetric(MultiTurnMetric):
    """Minimal MultiTurnMetric that returns a fixed score."""

    def init(self, run_config: RunConfig) -> None:  # type: ignore[override]
        return None

    async def _multi_turn_ascore(  # type: ignore[override]
        self,
        sample: MultiTurnSample,
        callbacks: t.Any = None,
    ) -> float:
        return 0.42


def _make_sample() -> MultiTurnSample:
    return MultiTurnSample(
        user_input=[AIMessage(content="hello")],
    )


@pytest.fixture
def captured_events(monkeypatch):
    """Patch _analytics_batcher with a recorder and yield the captured list."""
    captured: list = []

    class _FakeBatcher:
        def add_evaluation(self, event: EvaluationEvent) -> None:
            captured.append(event)

    monkeypatch.setattr("ragas.metrics.base._analytics_batcher", _FakeBatcher())
    return captured


def test_multi_turn_score_emits_multi_turn_evaluation_type(captured_events):
    """multi_turn_score must tag the analytics event as MULTI_TURN, not SINGLE_TURN."""
    metric = _StubMultiTurnMetric()
    sample = _make_sample()

    metric.multi_turn_score(sample)

    assert len(captured_events) == 1
    event = captured_events[0]
    assert isinstance(event, EvaluationEvent)
    assert event.evaluation_type == MetricType.MULTI_TURN.name
    assert event.metrics == [metric.name]


@pytest.mark.asyncio
async def test_multi_turn_ascore_emits_multi_turn_evaluation_type(captured_events):
    """multi_turn_ascore must tag the analytics event as MULTI_TURN, not SINGLE_TURN."""
    metric = _StubMultiTurnMetric()
    sample = _make_sample()

    await metric.multi_turn_ascore(sample)

    assert len(captured_events) == 1
    event = captured_events[0]
    assert isinstance(event, EvaluationEvent)
    assert event.evaluation_type == MetricType.MULTI_TURN.name
    assert event.metrics == [metric.name]

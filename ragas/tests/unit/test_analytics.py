from __future__ import annotations

import math
import time
import typing as t

import numpy as np
import pytest
from langchain_core.outputs import Generation, LLMResult
from langchain_core.prompt_values import StringPromptValue as PromptValue

from ragas._analytics import EvaluationEvent
from ragas.llms.base import BaseRagasLLM


class EchoLLM(BaseRagasLLM):
    def generate_text(  # type: ignore
        self,
        prompt: PromptValue,
    ) -> LLMResult:
        return LLMResult(generations=[[Generation(text=prompt.to_string())]])

    async def agenerate_text(  # type: ignore
        self,
        prompt: PromptValue,
    ) -> LLMResult:
        return LLMResult(generations=[[Generation(text=prompt.to_string())]])


def test_debug_tracking_flag():
    import os

    from ragas._analytics import RAGAS_DEBUG_TRACKING

    assert os.environ.get(RAGAS_DEBUG_TRACKING, "").lower() == "true"


def test_base_event():
    from ragas._analytics import BaseEvent

    be = BaseEvent(event_type="evaluation")
    assert isinstance(be.model_dump().get("event_type"), str)
    assert isinstance(be.model_dump().get("user_id"), str)


def test_evaluation_event():
    from ragas._analytics import EvaluationEvent

    evaluation_event = EvaluationEvent(
        event_type="evaluation",
        metrics=["harmfulness"],
        num_rows=1,
        language="english",
        evaluation_type="SINGLE_TURN",
    )

    payload = evaluation_event.model_dump()
    assert isinstance(payload.get("user_id"), str)
    assert isinstance(payload.get("evaluation_type"), str)
    assert isinstance(payload.get("metrics"), list)


def test_device_fingerprint_consistency():
    """Test that device fingerprint is consistent across multiple calls."""
    from ragas._analytics import _get_device_fingerprint, get_userid

    # Clear LRU cache to ensure fresh calls
    _get_device_fingerprint.cache_clear()
    get_userid.cache_clear()

    # Test device fingerprint consistency
    fp1 = _get_device_fingerprint()
    fp2 = _get_device_fingerprint()
    assert fp1 == fp2
    assert len(fp1) == 64  # SHA256 hash should be 64 chars

    # Test user ID consistency
    uid1 = get_userid()
    uid2 = get_userid()
    assert uid1 == uid2
    assert uid1.startswith("a-")
    assert len(uid1) == 34  # "a-" + 32 chars


def test_testset_generation_tracking(monkeypatch):
    import ragas._analytics as analyticsmodule
    from ragas._analytics import TestsetGenerationEvent, track
    from ragas.testset.synthesizers import default_query_distribution

    distributions = default_query_distribution(llm=EchoLLM())

    testset_event_payload = TestsetGenerationEvent(
        event_type="testset_generation",
        evolution_names=[e.name for e, _ in distributions],
        evolution_percentages=[p for _, p in distributions],
        num_rows=10,
        language="english",
    )

    assert testset_event_payload.model_dump()["evolution_names"] == [
        "single_hop_specifc_query_synthesizer",
        "multi_hop_abstract_query_synthesizer",
        "multi_hop_specific_query_synthesizer",
    ]

    assert all(
        np.isclose(
            testset_event_payload.model_dump()["evolution_percentages"],
            [
                0.33,
                0.33,
                0.33,
            ],
            atol=0.01,
        ).tolist()
    )

    # just in the case you actually want to check if tracking is working in the
    # dashboard
    if False:
        monkeypatch.setattr(analyticsmodule, "do_not_track", lambda: False)
        monkeypatch.setattr(analyticsmodule, "_usage_event_debugging", lambda: False)
        track(testset_event_payload)


def test_was_completed(monkeypatch):
    from ragas._analytics import IsCompleteEvent, track_was_completed

    event_properties_list: t.List[IsCompleteEvent] = []

    def echo_track(event_properties):
        event_properties_list.append(event_properties)

    monkeypatch.setattr("ragas._analytics.track", echo_track)

    @track_was_completed
    def test(raise_error=True):
        if raise_error:
            raise ValueError("test")
        else:
            pass

    with pytest.raises(ValueError):
        test(raise_error=True)

    assert event_properties_list[-1].event_type == "test"
    assert event_properties_list[-1].is_completed is False

    test(raise_error=False)

    assert event_properties_list[-1].event_type == "test"
    assert event_properties_list[-1].is_completed is True


evaluation_events_and_num_rows = [
    (  # 5 same events
        [
            EvaluationEvent(
                event_type="evaluation",
                metrics=["harmfulness"],
                num_rows=1,
                evaluation_type="SINGLE_TURN",
                language="english",
            )
            for _ in range(5)
        ],
        [5],
    ),
    (  # 5 different events with different metrics
        [
            EvaluationEvent(
                event_type="evaluation",
                metrics=[f"harmfulness_{i}"],
                num_rows=1,
                evaluation_type="SINGLE_TURN",
                language="english",
            )
            for i in range(5)
        ],
        [1, 1, 1, 1, 1],
    ),
    (  # 5 different events with different num_rows but 2 group of metrics
        [
            EvaluationEvent(
                metrics=["harmfulness"],
                num_rows=1,
                evaluation_type="SINGLE_TURN",
                language="english",
            )
            for i in range(10)
        ]
        + [
            EvaluationEvent(
                event_type="evaluation",
                metrics=["accuracy"],
                num_rows=1,
                evaluation_type="SINGLE_TURN",
                language="english",
            )
            for i in range(5)
        ],
        [10, 5],
    ),
]


@pytest.mark.parametrize(
    "evaluation_events, expected_num_rows_set", evaluation_events_and_num_rows
)
def test_analytics_batcher_join_evaluation_events(
    monkeypatch, evaluation_events, expected_num_rows_set
):
    """
    Test if the batcher joins the evaluation events correctly
    """
    from ragas._analytics import AnalyticsBatcher

    batcher = AnalyticsBatcher()

    joined_events = batcher._join_evaluation_events(evaluation_events)
    assert len(joined_events) == len(expected_num_rows_set)
    assert sorted(e.num_rows for e in joined_events) == sorted(expected_num_rows_set)


@pytest.mark.skip(reason="This test is flaky and needs to be fixed")
@pytest.mark.parametrize(
    "evaluation_events, expected_num_rows_set", evaluation_events_and_num_rows
)
def test_analytics_batcher_flush(monkeypatch, evaluation_events, expected_num_rows_set):
    """
    Test if the batcher flushes the events correctly
    """
    from ragas._analytics import AnalyticsBatcher

    FLUSH_INTERVAL = 0.3
    BATCH_SIZE = 5
    batcher = AnalyticsBatcher(batch_size=BATCH_SIZE, flush_interval=FLUSH_INTERVAL)

    # Use a list to hold the counter so it can be modified in the nested function
    flush_mock_call_count = [0]

    def flush_mock():
        # Access the list and modify its first element
        flush_mock_call_count[0] += 1
        batcher.buffer = []
        batcher.last_flush_time = time.time()

    monkeypatch.setattr(batcher, "flush", flush_mock)

    for event in evaluation_events[:-1]:
        batcher.add_evaluation(event)

    # Access the counter using flush_mock_call_count[0]
    time.sleep(FLUSH_INTERVAL + 0.1)
    batcher.add_evaluation(evaluation_events[-1])
    assert flush_mock_call_count[0] == math.ceil(
        sum(expected_num_rows_set) / BATCH_SIZE
    )

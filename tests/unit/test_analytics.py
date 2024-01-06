from __future__ import annotations


from ragas._analytics import EvaluationEvent


def test_add_userid():
    evaluation_event = EvaluationEvent(
        event_type="evaluation", metrics=["harmfulness"], num_rows=1, evaluation_mode=""
    )
    payload = evaluation_event.__dict__
    assert payload.get("user_id") is not None

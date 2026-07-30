"""Tests for the legacy InstanceRubrics metric in
src/ragas/metrics/_instance_specific_rubrics.py.

Covers prompt-construction edge cases that the modern collections
metric (tests/unit/test_instance_specific_rubrics_collections.py) does not
exercise, specifically the user_input=None case which historically
caused the literal string "None" to be formatted into the LLM prompt.
"""

import pytest

from ragas.metrics._domain_specific_rubrics import ScoreFeedback
from ragas.metrics._instance_specific_rubrics import InstanceRubrics


class _FakeLLM:
    """Stand-in for a BaseRagasLLM / InstructorBaseRagasLLM.

    The metric only needs `self.llm` to be truthy for the assertion
    at the top of `_ascore`. The actual LLM call is monkey-patched
    on the prompt object inside the test, so this class never
    needs to talk to a real model.
    """


@pytest.fixture
def sample_rubrics():
    return {
        "score1_description": "The response is completely off-topic",
        "score2_description": "The response is partially relevant",
        "score3_description": "The response addresses the topic but lacks depth",
        "score4_description": "The response is good with minor improvements needed",
        "score5_description": "The response is excellent and comprehensive",
    }


@pytest.mark.asyncio
async def test_user_input_none_with_contexts_does_not_emit_none(sample_rubrics):
    """Regression test: passing user_input=None together with
    retrieved_contexts must not interpolate the literal string "None"
    into the LLM prompt.
    """
    metric = InstanceRubrics(llm=_FakeLLM())

    captured: dict = {}

    async def _fake_generate(data, llm, callbacks):  # mirrors PydanticPrompt.generate
        captured["data"] = data
        return ScoreFeedback(feedback="ok", score=5)

    metric.single_turn_prompt.generate = _fake_generate  # type: ignore[method-assign]

    row = {
        "user_input": None,
        "retrieved_contexts": ["ctx-1", "ctx-2"],
        "response": "some response",
        "reference": None,
        "rubrics": sample_rubrics,
    }

    await metric._ascore(row=row, callbacks=[])

    assert "data" in captured, "prompt.generate was not called"
    prompt_input = captured["data"]
    # The bug: the formatted user_input contained the word "None".
    assert prompt_input.user_input is not None
    assert "None" not in prompt_input.user_input
    # The fixed prompt should still mention the context the user supplied.
    assert "ctx-1" in prompt_input.user_input
    assert "ctx-2" in prompt_input.user_input


@pytest.mark.asyncio
async def test_user_input_string_with_contexts_unchanged(sample_rubrics):
    """Sanity check: when user_input is a non-empty string and contexts
    are provided, the prompt is unchanged from the previous behaviour
    (i.e., "answer using context: ..." appended to the user input).
    """
    metric = InstanceRubrics(llm=_FakeLLM())

    captured: dict = {}

    async def _fake_generate(data, llm, callbacks):
        captured["data"] = data
        return ScoreFeedback(feedback="ok", score=5)

    metric.single_turn_prompt.generate = _fake_generate  # type: ignore[method-assign]

    row = {
        "user_input": "What is X?",
        "retrieved_contexts": ["ctx-1"],
        "response": "an answer",
        "reference": None,
        "rubrics": sample_rubrics,
    }

    await metric._ascore(row=row, callbacks=[])

    prompt_input = captured["data"]
    assert "What is X? answer using context: ctx-1" in prompt_input.user_input

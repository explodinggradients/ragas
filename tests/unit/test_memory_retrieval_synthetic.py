"""Synthetic tests for agent memory retrieval evaluation."""

from __future__ import annotations

import pytest

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import IDBasedContextRecall, NonLLMContextRecall

SYNTHETIC_MEMORY_SAMPLE = SingleTurnSample(
    user_input="What meeting time does the user prefer?",
    reference_contexts=[
        "User prefers morning meetings before 10am.",
        "User avoids back-to-back meetings on Fridays.",
    ],
    retrieved_contexts=[
        "User prefers morning meetings before 10am.",
        "User works in the Pacific timezone.",
    ],
)

SYNTHETIC_MEMORY_IDS = SingleTurnSample(
    user_input="What dietary restrictions should we remember?",
    reference_context_ids=["mem-veg", "mem-peanut-allergy"],
    retrieved_context_ids=["mem-veg", "mem-oat-milk"],
)


@pytest.mark.asyncio
async def test_non_llm_context_recall_partial_memory_match():
    metric = NonLLMContextRecall()
    score = await metric.single_turn_ascore(SYNTHETIC_MEMORY_SAMPLE)
    assert score == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_non_llm_context_recall_perfect_memory_match():
    metric = NonLLMContextRecall()
    sample = SingleTurnSample(
        user_input="What is the user's project focus?",
        reference_contexts=[
            "User is building a customer support agent.",
            "Launch target is Q3.",
        ],
        retrieved_contexts=[
            "User is building a customer support agent.",
            "Launch target is Q3.",
        ],
    )
    score = await metric.single_turn_ascore(sample)
    assert score == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_id_based_context_recall_for_memory_ids():
    metric = IDBasedContextRecall()
    score = await metric.single_turn_ascore(SYNTHETIC_MEMORY_IDS)
    assert score == pytest.approx(0.5)
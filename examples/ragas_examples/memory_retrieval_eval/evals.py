"""Evaluate long-term memory retrieval with synthetic agent memory data.

This example scores how well a memory retriever surfaces facts that an agent
should recall. It uses only synthetic data and a deterministic numeric metric,
so no API keys are required.
"""

from __future__ import annotations

import asyncio
from typing import List

from ragas import Dataset, experiment
from ragas.metrics.numeric import numeric_metric
from ragas.metrics.result import MetricResult


def _normalize_memory_items(items: List[str]) -> List[str]:
    return [item.strip().lower() for item in items if item and item.strip()]


@numeric_metric(name="memory_recall_overlap", allowed_values=(0.0, 1.0))
def memory_recall_overlap(expected_memory: List[str], retrieved_memory: List[str]):
    """Fraction of expected memory facts matched in retrieved memory."""
    expected = _normalize_memory_items(expected_memory)
    retrieved = _normalize_memory_items(retrieved_memory)

    if not expected:
        return MetricResult(value=1.0, reason="No expected memory items to match")

    matched = 0
    for fact in expected:
        if any(fact in candidate or candidate in fact for candidate in retrieved):
            matched += 1

    score = matched / len(expected)
    return MetricResult(
        value=score,
        reason=f"Matched {matched}/{len(expected)} expected memory facts",
    )


def load_dataset() -> Dataset:
    """Build a tiny synthetic dataset for long-term memory retrieval."""
    dataset = Dataset(
        name="memory_retrieval_synthetic",
        backend="local/csv",
        root_dir=".",
    )

    rows = [
        {
            "user_input": "What meeting time does the user prefer?",
            "expected_memory": [
                "User prefers morning meetings before 10am.",
                "User avoids back-to-back meetings on Fridays.",
            ],
            "retrieved_memory": [
                "User prefers morning meetings before 10am.",
                "User works in the Pacific timezone.",
            ],
        },
        {
            "user_input": "What dietary restrictions should we remember?",
            "expected_memory": [
                "User is vegetarian.",
                "User is allergic to peanuts.",
            ],
            "retrieved_memory": [
                "User is vegetarian.",
                "User drinks oat milk in coffee.",
            ],
        },
        {
            "user_input": "Summarize the user's project context.",
            "expected_memory": [
                "User is building a customer support agent.",
                "Launch target is Q3.",
            ],
            "retrieved_memory": [
                "User is building a customer support agent.",
                "Launch target is Q3.",
                "Team uses Python and FastAPI.",
            ],
        },
    ]

    for row in rows:
        dataset.append(row)

    dataset.save()
    return dataset


@experiment()
async def run_experiment(row):
    score = memory_recall_overlap.score(
        expected_memory=row["expected_memory"],
        retrieved_memory=row["retrieved_memory"],
    )

    return {
        "user_input": row["user_input"],
        "expected_memory": row["expected_memory"],
        "retrieved_memory": row["retrieved_memory"],
        "memory_recall_overlap": score.value,
        "memory_recall_reason": score.reason,
    }


async def main():
    dataset = load_dataset()
    results = await run_experiment.arun(dataset)
    print("Memory retrieval evaluation results:")
    print(results)


if __name__ == "__main__":
    asyncio.run(main())
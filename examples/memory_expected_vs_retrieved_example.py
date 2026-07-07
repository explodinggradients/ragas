#!/usr/bin/env python3
"""Compare expected memory against retrieved memory using Ragas metrics.

This standalone script maps agent-memory evaluation onto Ragas evaluation
fields:

- ``reference_contexts`` → expected memory facts the agent should recall
- ``retrieved_contexts`` → memory chunks returned by the retriever

It uses ``NonLLMContextRecall``, so no API keys are required.
"""

from __future__ import annotations

import asyncio

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import NonLLMContextRecall

SYNTHETIC_MEMORY_CASES = [
    {
        "user_input": "What timezone is the user in?",
        "reference_contexts": [
            "User works in the Pacific timezone.",
            "User schedules calls after 9am local time.",
        ],
        "retrieved_contexts": [
            "User works in the Pacific timezone.",
            "User prefers async updates over live meetings.",
        ],
    },
    {
        "user_input": "What are the user's food preferences?",
        "reference_contexts": [
            "User is vegetarian.",
            "User avoids spicy food.",
        ],
        "retrieved_contexts": [
            "User is vegetarian.",
            "User likes Italian cuisine.",
        ],
    },
]


async def score_memory_retrieval(sample: dict) -> float:
    metric = NonLLMContextRecall()
    turn = SingleTurnSample(
        user_input=sample["user_input"],
        reference_contexts=sample["reference_contexts"],
        retrieved_contexts=sample["retrieved_contexts"],
    )
    return await metric.single_turn_ascore(turn)


async def main() -> None:
    print("Expected memory vs retrieved memory (synthetic data)\n")

    scores = []
    for case in SYNTHETIC_MEMORY_CASES:
        score = await score_memory_retrieval(case)
        scores.append(score)
        print(f"Query: {case['user_input']}")
        print(f"  Expected memory: {case['reference_contexts']}")
        print(f"  Retrieved memory: {case['retrieved_contexts']}")
        print(f"  NonLLM context recall: {score:.2f}\n")

    average = sum(scores) / len(scores)
    print(f"Average memory recall score: {average:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
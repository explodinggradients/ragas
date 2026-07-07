# Retrieval Drift Regression Pattern

Memory and RAG retrievers can silently degrade when embeddings, chunking, or
index content change. This guide shows a lightweight regression pattern using
Ragas and a fixed synthetic dataset.

## When to use this pattern

- You changed the embedding model or chunk size
- You updated the memory schema or indexing pipeline
- You refactored retrieval ranking logic
- You want a CI guardrail before shipping retriever changes

## Regression skeleton

1. **Freeze a synthetic benchmark** with queries and expected memory facts.
2. **Record a baseline** recall/precision score from the current retriever.
3. **Re-run after changes** with the same dataset and metrics.
4. **Fail the build** when scores drop beyond a tolerance.

```python
import asyncio

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import NonLLMContextRecall

BENCHMARK = [
    {
        "user_input": "What meeting time does the user prefer?",
        "reference_contexts": [
            "User prefers morning meetings before 10am.",
            "User avoids back-to-back meetings on Fridays.",
        ],
        "retrieved_contexts": [
            "User prefers morning meetings before 10am.",
            "User works in the Pacific timezone.",
        ],
    }
]

BASELINE_RECALL = 0.50
TOLERANCE = 0.10


async def evaluate_retrieval(cases: list) -> float:
    metric = NonLLMContextRecall()
    scores = []
    for case in cases:
        sample = SingleTurnSample(**case)
        scores.append(await metric.single_turn_ascore(sample))
    return sum(scores) / len(scores)


async def main():
    score = await evaluate_retrieval(BENCHMARK)
    if score < BASELINE_RECALL - TOLERANCE:
        raise AssertionError(
            f"Retrieval drift detected: score={score:.2f}, "
            f"baseline={BASELINE_RECALL:.2f}"
        )
    print(f"Retrieval regression passed: {score:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
```

## Choosing metrics

| Scenario | Metric | Why |
| --- | --- | --- |
| Labeled expected memory | `NonLLMContextRecall` | Fast, deterministic, no API keys |
| Ranked memory chunks | `NonLLMContextPrecision` | Detects noisy or stale chunks rising in rank |
| Stable memory IDs | `IDBasedContextRecall` | Compares IDs instead of text content |

## CI integration tips

- Keep the benchmark small (5–20 rows) so CI stays fast.
- Store expected scores in code or a versioned JSON file.
- Log per-row failures to identify which memory fact was missed.
- Re-baseline intentionally when retrieval behavior should change.

## Related guides

- [Context Recall](../../concepts/metrics/available_metrics/context_recall.md)
- [Context Precision](../../concepts/metrics/available_metrics/context_precision.md)
- [Evaluation Sample](../../concepts/components/eval_sample.md)
# Context Relevance for Agent Memory

Agent memory systems store facts across sessions and retrieve them when a new
query arrives. Ragas retrieval metrics apply directly to this pattern when you
map memory fields onto evaluation samples.

## Field mapping

| Agent memory concept | Ragas field | Purpose |
| --- | --- | --- |
| User query | `user_input` | The question that triggers memory retrieval |
| Expected memory facts | `reference_contexts` | Facts the agent should recall for the query |
| Retrieved memory chunks | `retrieved_contexts` | Chunks returned by the memory retriever |
| Final agent answer | `response` | Optional answer grounded in retrieved memory |
| Gold answer | `reference` | Optional target answer for end-to-end checks |

Use `reference_contexts` when you know which stored facts are relevant.
Use `retrieved_contexts` to capture what the retriever actually returned.

## Recommended metrics

### Memory recall (coverage)

Use [Context Recall](../../concepts/metrics/available_metrics/context_recall.md)
or `NonLLMContextRecall` when you have labeled expected memory facts and want
to measure how many were retrieved.

`NonLLMContextRecall` is useful for synthetic regression tests because it does
not require an LLM API key.

### Memory precision (signal-to-noise)

Use [Context Precision](../../concepts/metrics/available_metrics/context_precision.md)
when retrieved memory may include stale or unrelated chunks. Precision checks
whether relevant memory appears before irrelevant memory in the ranked results.

### Answer grounding

If the agent produces a final response, add
[Faithfulness](../../concepts/metrics/available_metrics/faithfulness.md) to
verify the answer stays within retrieved memory.

## Synthetic evaluation loop

1. Define a query and the expected memory facts (`reference_contexts`).
2. Run your retriever and record `retrieved_contexts`.
3. Score recall and precision with Ragas metrics.
4. Track scores over time to catch retrieval drift.

```python
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import NonLLMContextRecall

sample = SingleTurnSample(
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

metric = NonLLMContextRecall()
score = await metric.single_turn_ascore(sample)
```

## Practical tips

- Keep each memory chunk atomic so recall failures are easy to debug.
- Store memory IDs when possible and use `IDBasedContextRecall` for fast
  regression checks.
- Separate retrieval evaluation from answer evaluation. Poor recall usually
  caps downstream answer quality.
- Start with synthetic datasets before collecting production traces.

## Related guides

- [Evaluation Sample](../../concepts/components/eval_sample.md)
- [Context Recall](../../concepts/metrics/available_metrics/context_recall.md)
- [Context Precision](../../concepts/metrics/available_metrics/context_precision.md)
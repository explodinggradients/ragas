## `QuotedSpansAlignment`

**What:** A metric that measures the fraction of quoted spans in a model's answer
that appear verbatim in the retrieved sources.  The score is in the range
[0, 1], where 1.0 indicates every quoted span is supported by evidence and 0.0
indicates no quoted spans are found in the sources.

**Why:** Users place extra trust in exact quotes.  When a model quotes facts
that aren't present in its evidence, it undermines reliability.  This metric
helps catch cases of citation drift where quoted phrases in the answer are
unsupported.

## Modern Collections API (Recommended)

```python
from ragas.metrics.collections import QuotedSpansAlignment

metric = QuotedSpansAlignment()

result = await metric.ascore(
    response='The study found that "machine learning improves accuracy".',
    retrieved_contexts=["Machine learning improves accuracy by 15%."]
)
print(f"Score: {result.value}")  # 1.0
print(f"Reason: {result.reason}")  # "Matched 1/1 quoted spans"
```

**Parameters:**

- `name`: The metric name (default: "quoted_spans_alignment")
- `casefold`: Whether to normalize text by lower-casing before matching (default: True)
- `min_span_words`: Minimum number of words in a quoted span (default: 3)

**Input:**

- `response: str` – the model's response containing quoted spans
- `retrieved_contexts: List[str]` – list of source passages to check against

**Output:** A `MetricResult` with:

- `value`: Score in [0, 1]
- `reason`: Description of matched/total spans

**Notes:**

- The implementation normalizes text by collapsing whitespace and lower‑casing.
- Spans shorter than three words are ignored by default; adjust `min_span_words` to change this.
- If no quoted spans are found in the response, the score is 1.0 (nothing to verify).

---


## `citation_alignment_quoted_spans`

**What:** A metric that measures the fraction of quoted spans in a model’s answer
that appear verbatim in the retrieved sources.  The score is in the range
[0, 1], where 1.0 indicates every quoted span is supported by evidence and 0.0
indicates no quoted spans are found in the sources.

**Why:** Users place extra trust in exact quotes.  When a model quotes facts
that aren’t present in its evidence, it undermines reliability.  This metric
helps catch cases of citation drift where quoted phrases in the answer are
unsupported.

**Input shape:**

- `answers: List[str]` – list of model answers (length N)
- `sources: List[List[str]]` – list (length N) of lists of source passages

**Output:** A dictionary containing:

```python
{
  "citation_alignment_quoted_spans": float,  # score in [0,1]
  "matched": float,                          # number of spans found in sources
  "total": float                            # total number of spans considered
}
```

**Notes:**

- The implementation normalizes text by collapsing whitespace and lower‑casing.
- Spans shorter than three words are ignored by default; adjust `min_len` to change this.
- If no quoted spans are found across all answers, the score is defined as 0.0 with
  `total = 0`.
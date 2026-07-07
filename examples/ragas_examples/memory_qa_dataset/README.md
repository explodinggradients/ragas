# Memory QA Synthetic Dataset

A tiny synthetic dataset for evaluating agent memory retrieval with Ragas.

## Columns

| Column | Description |
| --- | --- |
| `user_input` | Query that triggers memory retrieval |
| `reference_contexts` | Expected memory facts (pipe-separated in CSV) |
| `retrieved_contexts` | Memory chunks returned by the retriever |
| `reference` | Optional gold answer for end-to-end checks |

## Usage

```python
from ragas.metrics import NonLLMContextRecall
from ragas_examples.memory_qa_dataset.load_dataset import load_memory_qa_dataset

dataset = load_memory_qa_dataset()
result = evaluate(dataset=dataset, metrics=[NonLLMContextRecall()])
```

Run the loader directly:

```bash
python -m ragas_examples.memory_qa_dataset.load_dataset
```

All rows are fully synthetic and safe for CI regression tests.
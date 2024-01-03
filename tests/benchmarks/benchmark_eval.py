import time

from datasets import DatasetDict, load_dataset

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    context_recall,
    answer_correctness,
    context_relevancy,
    context_precision,
    context_utilization,
    answer_similarity,
)
from ragas.metrics.critique import harmfulness

# data
ds = load_dataset("explodinggradients/fiqa", "ragas_eval")
assert isinstance(ds, DatasetDict)
fiqa = ds["baseline"]

# metrics
metrics = [
    faithfulness,
    context_recall,
    answer_correctness,
    harmfulness,
    context_relevancy,
    context_precision,
    context_utilization,
    answer_similarity,
]

if __name__ == "__main__":
    # asyncio
    start = time.time()
    print("ignored")
    # _ = evaluate(
    #     fiqa,
    #     metrics=[
    #         faithfulness,
    #     ],
    #     is_async=True,
    # )
    print(f"Time taken [Asyncio]: {time.time() - start:.2f}s")

    # Threads
    start = time.time()
    _ = evaluate(
        fiqa,
        metrics=metrics,
        is_async=False,
    )
    print(f"Time taken [Threads]: {time.time() - start:.2f}s")

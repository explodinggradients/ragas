import time

from datasets import DatasetDict, load_dataset

from ragas import evaluate
from ragas.metrics import faithfulness

# data
ds = load_dataset("explodinggradients/fiqa", "ragas_eval")
assert isinstance(ds, DatasetDict)
fiqa = ds["baseline"]

if __name__ == "__main__":
    # asyncio
    start = time.time()
    _ = evaluate(
        fiqa,
        metrics=[
            faithfulness,
        ],
        is_async=True,
    )
    print(f"Time taken [Asyncio]: {time.time() - start:.2f}s")

    # Threads
    start = time.time()
    _ = evaluate(
        fiqa,
        metrics=[
            faithfulness,
        ],
        is_async=False,
    )
    print(f"Time taken [Threads]: {time.time() - start:.2f}s")

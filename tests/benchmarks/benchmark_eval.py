import time

from ragas import evaluate
from ragas.metrics import (
    ContextUtilization,
    answer_correctness,
    answer_relevancy,
    answer_similarity,
    context_precision,
    context_recall,
    faithfulness,
)

from ..e2e.test_dataset_utils import load_amnesty_dataset_safe

# from ragas.metrics.critique import harmfulness  # Import unavailable

# data - using safe dataset loading
eval_dataset = load_amnesty_dataset_safe("english_v2")

# metrics
metrics = [
    faithfulness,
    context_recall,
    answer_relevancy,
    answer_correctness,
    context_precision,
    ContextUtilization(),
    answer_similarity,
]

# os.environ["PYTHONASYNCIODEBUG"] = "1"
IGNORE_ASYNCIO = False

if __name__ == "__main__":
    # asyncio
    print("Starting [Asyncio]")
    start = time.time()
    _ = evaluate(
        eval_dataset,
        metrics=metrics,
    )
    print(f"Time taken [Asyncio]: {time.time() - start:.2f}s")

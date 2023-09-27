from datasets import load_dataset
from torch.cuda import is_available

from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from ragas.metrics.critique import harmfulness

DEVICE = "cuda" if is_available() else "cpu"

# data
ds = load_dataset("explodinggradients/fiqa", "ragas_eval")["baseline"]

if __name__ == "__main__":
    result = evaluate(
        ds.select(range(5)),
        metrics=[
            answer_relevancy,
            context_precision,
            faithfulness,
            harmfulness,
            context_recall,
        ],
    )
    print(result)

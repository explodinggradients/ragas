import os

from datasets import Dataset, load_dataset
from torch.cuda import is_available

from ragas import evaluate
from ragas.metrics import answer_relevancy, context_relevancy, faithfulness

DEVICE = "cuda" if is_available() else "cpu"

# data
ds = load_dataset("explodinggradients/fiqa", "ragas_eval")["baseline"]

if __name__ == "__main__":
    result = evaluate(
        ds,
        metrics=[answer_relevancy, context_relevancy, faithfulness],
    )
    print(result)

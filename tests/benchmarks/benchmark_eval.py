import os

from datasets import Dataset, load_dataset
from torch.cuda import is_available

from ragas import evaluate
from ragas.metrics import answer_relevancy, context_relevancy, faithfulness

DEVICE = "cuda" if is_available() else "cpu"

PATH_TO_DATSET_GIT_REPO = "../../../datasets/fiqa/"
dataset_dir = os.environ.get("DATASET_DIR", PATH_TO_DATSET_GIT_REPO)
if os.path.isdir(dataset_dir):
    ds = Dataset.from_csv(os.path.join(dataset_dir, "baseline.csv"))
    assert isinstance(ds, Dataset)
else:
    # data
    ds = load_dataset("explodinggradients/fiqa", "ragas_eval")["baseline"]

if __name__ == "__main__":
    result = evaluate(
        ds,
        metrics=[answer_relevancy, context_relevancy, faithfulness],
    )
    print(result)

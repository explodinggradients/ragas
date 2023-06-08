import os

from datasets import Dataset
from torch.cuda import is_available

from ragas import evaluate
from ragas.metrics import answer_relevancy, context_relavancy, factuality

DEVICE = "cuda" if is_available() else "cpu"

PATH_TO_DATSET_GIT_REPO = "../../../datasets/fiqa/"
assert os.path.isdir(PATH_TO_DATSET_GIT_REPO), "Dataset not found"
ds = Dataset.from_json(os.path.join(PATH_TO_DATSET_GIT_REPO, "gen_ds.json"))
assert isinstance(ds, Dataset)

if __name__ == "__main__":
    result = evaluate(
        ds,
        metrics=[answer_relevancy, context_relavancy, factuality],
    )
    print(result)

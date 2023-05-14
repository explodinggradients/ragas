import typing as t

from datasets import Dataset, load_dataset
from torch.cuda import is_available
from tqdm import tqdm
from utils import print_table, timeit

from ragas.metrics import (
    EditDistance,
    EditRatio,
    EntailmentScore,
    Evaluation,
    Rouge1,
    Rouge2,
    RougeL,
    SBERTScore,
)

DEVICE = "cuda" if is_available() else "cpu"
BATCHES = [0, 1]
# init metrics
sbert_score = SBERTScore(similarity_metric="cosine")
entail = EntailmentScore(max_length=512, device=DEVICE)
METRICS = {
    "Rouge1": Rouge1,
    "Rouge2": Rouge2,
    "RougeL": RougeL,
    "EditRatio": EditRatio,
    "EditDistance": EditDistance,
    # "SBERTScore": sbert_score,
    # "EntailmentScore": entail,
}
DS = load_dataset("explodinggradients/eli5-test", split="test_eli5")


def setup() -> t.Iterator[tuple[str, Evaluation, Dataset]]:
    metrics = [m for m in METRICS.values()]
    for b in BATCHES:
        setup_name = f"batch-{b}"
        assert isinstance(DS, Dataset), f"{type(DS)} found in the place of Dataset!"
        batched = False if b == 0 else True
        e = Evaluation(
            metrics=metrics,
            batched=batched,
            batch_size=b,
        )
        yield setup_name, e, DS


@timeit
def evaluate(e: Evaluation, ds: Dataset):
    e.eval(ds["ground_truth"], ds["generated_text"])


if __name__ == "__main__":
    results = {}
    for setup_name, e, ds in tqdm(setup(), total=len(BATCHES)):
        mean, var = evaluate(e, ds)
        results[setup_name] = (mean, var)

    print_table(results)

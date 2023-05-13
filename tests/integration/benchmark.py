import typing as t
from dataclasses import dataclass

from datasets import Dataset, load_dataset
from tqdm import tqdm
from utils import timeit

from belar.metrics import (
    EditDistance,
    EditRatio,
    EntailmentScore,
    Evaluation,
    Rouge1,
    Rouge2,
    RougeL,
    SBERTScore,
)

DEVICE = ("cuda",)
BATCHES = [0, 1, 10, 20, 30, 60]
# init metrics
sbert_score = SBERTScore(similarity_metric="cosine")
entail = EntailmentScore(max_length=512)
METRICS = {
    "Rouge1": Rouge1,
    "Rouge2": Rouge2,
    "RougeL": RougeL,
    "EditRatio": EditRatio,
    "EditDistance": EditDistance,
}


@dataclass
class BenchmarkConfig:
    device: t.Literal["cpu", "cuda"]
    batch_sizes: list[int]
    metrics: list[str]


def setup() -> t.Iterator[tuple[str, Evaluation, Dataset]]:
    metrics = [m for m in METRICS.values()]
    for b in BATCHES:
        setup_name = f"batch-{b}"
        ds = load_dataset("explodinggradients/eli5-test", split="test_eli5")
        assert isinstance(ds, Dataset), f"{type(ds)} found in the place of Dataset!"
        batched = False if b == 0 else True
        e = Evaluation(
            metrics=metrics,
            batched=batched,
            batch_size=b,
        )
        yield setup_name, e, ds


@timeit
def evaluate(e: Evaluation, ds: Dataset):
    e.eval(ds["ground_truth"], ds["generated_text"])


if __name__ == "__main__":
    results = {}
    for setup_name, e, ds in tqdm(setup()):
        mean, var = evaluate(e, ds)
        results[setup_name] = (mean, var)

    print(results)

import typing as t

from datasets import Dataset, arrow_dataset, load_dataset
from torch.cuda import is_available
from tqdm import tqdm
from utils import print_table, timeit

from ragas.metrics import (
    Evaluation,
    edit_distance,
    edit_ratio,
    q_square,
    rouge1,
    rouge2,
    rougeL,
)

DEVICE = "cuda" if is_available() else "cpu"
BATCHES = [0, 1]

METRICS = {
    "Rouge1": rouge1,
    "Rouge2": rouge2,
    "RougeL": rougeL,
    "EditRatio": edit_ratio,
    "EditDistance": edit_distance,
    # "SBERTScore": bert_score,
    # "EntailmentScore": entailment_score,
    "Qsquare": q_square,
}
DS = load_dataset("explodinggradients/eli5-test", split="test_eli5")
assert isinstance(DS, arrow_dataset.Dataset), "Not an arrow_dataset"
DS = DS.select(range(100))


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

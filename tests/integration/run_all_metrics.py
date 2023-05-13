from datasets import concatenate_datasets, load_dataset

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

ds = load_dataset("explodinggradients/eli5-test", split="test_eli5")
print(ds.shape)
sbert_score = SBERTScore(similarity_metric="cosine")
entail = EntailmentScore(max_length=512)

e = Evaluation(
    metrics=[Rouge1, Rouge2, RougeL, sbert_score, EditDistance, EditRatio, entail],
    batched=False,
    batch_size=30,
)
r = e.eval(ds["ground_truth"], ds["generated_text"])
print(r)
print(r.describe())

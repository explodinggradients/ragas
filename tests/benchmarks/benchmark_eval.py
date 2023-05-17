from datasets import arrow_dataset, load_dataset
from torch.cuda import is_available

from ragas.metrics import Evaluation, bert_score, edit_ratio, rougeL
from ragas.metrics.factual import EntailmentScore

DEVICE = "cuda" if is_available() else "cpu"
entailment_score = EntailmentScore(device=DEVICE, batch_size=2)
# q_square = Qsquare(device=DEVICE, batch_size=2)

DS = load_dataset("explodinggradients/ragas-webgpt", split="train")
assert isinstance(DS, arrow_dataset.Dataset), "Not an arrow_dataset"
DS = DS.select(range(500))

if __name__ == "__main__":
    e = Evaluation(
        metrics=[rougeL, edit_ratio, bert_score, entailment_score],
        batched=True,
        batch_size=64,
    )
    result = e.eval(DS["ground_truth"], DS["generated_text"])
    print(result)

from datasets import load_dataset

from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, faithfulness
from ragas.metrics.critique import harmfulness


def test_evaluate_e2e():
    ds = load_dataset("explodinggradients/fiqa", "ragas_eval")["baseline"]
    result = evaluate(
        ds.select(range(3)),
        metrics=[answer_relevancy, context_precision, faithfulness, harmfulness],
    )
    assert result is not None

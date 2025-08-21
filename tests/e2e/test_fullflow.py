import typing as t

from datasets import load_dataset

from ragas import EvaluationDataset, evaluate
from ragas.metrics import answer_relevancy, context_precision, faithfulness
from ragas.metrics._aspect_critic import harmfulness

if t.TYPE_CHECKING:
    from datasets import Dataset


def test_evaluate_e2e():
    ds = load_dataset("explodinggradients/amnesty_qa", "english_v3")["eval"]  # type: ignore
    result = evaluate(
        EvaluationDataset.from_hf_dataset(t.cast("Dataset", ds))[:1],
        metrics=[answer_relevancy, context_precision, faithfulness, harmfulness],
        show_progress=False,
    )
    assert result is not None

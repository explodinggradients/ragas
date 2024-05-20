import pytest
from datasets import load_dataset

from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)

# loading the V2 dataset
amnesty_qa = load_dataset("explodinggradients/amnesty_qa", "english_v2")["eval"]


@pytest.mark.ragas_ci
def test_amnesty_e2e():
    result = evaluate(
        amnesty_qa,
        metrics=[answer_relevancy, faithfulness, context_recall, context_precision],
        in_ci=True,
    )
    print(result)
    assert False

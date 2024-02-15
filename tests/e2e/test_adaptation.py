from ragas import adapt
from ragas.metrics import context_recall


def test_adapt():
    adapt([context_recall], language="spanish")
    assert context_recall.context_recall_prompt.language == "spanish"

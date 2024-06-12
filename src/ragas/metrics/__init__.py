from datasets import Dataset

from ragas.metrics._answer_correctness import AnswerCorrectness, answer_correctness
from ragas.metrics._answer_relevance import AnswerRelevancy, answer_relevancy
from ragas.metrics._answer_similarity import AnswerSimilarity, answer_similarity
from ragas.metrics._context_entities_recall import (
    ContextEntityRecall,
    context_entity_recall,
)
from ragas.metrics._context_precision import (
    ContextPrecision,
    ContextUtilization,
    context_precision,
    context_utilization,
)
from ragas.metrics._context_recall import ContextRecall, context_recall
from ragas.metrics._context_relevancy import ContextRelevancy, context_relevancy
from ragas.metrics._faithfulness import Faithfulness, faithfulness
from ragas.metrics.critique import AspectCritique
from ragas.metrics.base import Metric
from ragas.validation import validate_evaluation_modes

ALL_METRICS = [
    answer_correctness,
    faithfulness,
    answer_similarity,
    context_precision,
    context_utilization,
    context_recall,
    context_relevancy,
    answer_relevancy,
    context_entity_recall,
]


def get_available_metrics(ds: Dataset) -> list[Metric]:
    """
    Get the available metrics for the given dataset.
    E.g. if the dataset contains ("question", "answer", "contexts") columns, 
    the available metrics are those that can be evaluated in [qa, qac, qc] mode.
    """
    available_metrics = []
    for metric in ALL_METRICS:
        try:
            validate_evaluation_modes(ds, [metric])
            available_metrics.append(metric)
        except ValueError:
            continue
    return available_metrics


__all__ = [
    "AnswerCorrectness",
    "answer_correctness",
    "Faithfulness",
    "faithfulness",
    "AnswerSimilarity",
    "answer_similarity",
    "ContextPrecision",
    "context_precision",
    "ContextUtilization",
    "context_utilization",
    "ContextRecall",
    "context_recall",
    "AspectCritique",
    "context_relevancy",
    "ContextRelevancy",
    "AnswerRelevancy",
    "answer_relevancy",
    "ContextEntityRecall",
    "context_entity_recall",
    "get_available_metrics",
]

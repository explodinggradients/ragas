from datasets import Dataset

from ragas.metrics._answer_correctness import answer_correctness
from ragas.metrics._answer_relevance import answer_relevancy
from ragas.metrics._answer_similarity import answer_similarity
from ragas.metrics._context_entities_recall import context_entity_recall
from ragas.metrics._context_precision import context_precision, context_utilization
from ragas.metrics._context_recall import context_recall
from ragas.metrics._faithfulness import faithfulness
from ragas.metrics._noise_sensitivity import noise_sensitivity_relevant, noise_sensitivity_irrelevant
from ragas.metrics._rubrics_based import labelled_rubrics_score, reference_free_rubrics_score
from ragas.metrics.base import Metric
from ragas.validation import validate_evaluation_modes

ALL_METRICS = [
    answer_correctness,
    faithfulness,
    answer_similarity,
    context_precision,
    context_utilization,
    context_recall,
    answer_relevancy,
    context_entity_recall,
    noise_sensitivity_relevant,
    noise_sensitivity_irrelevant,
    labelled_rubrics_score,
    reference_free_rubrics_score,
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

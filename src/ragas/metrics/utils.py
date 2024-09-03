from ragas.dataset_schema import EvaluationDataset
from ragas.metrics import ALL_METRICS
from ragas.metrics.base import Metric
from ragas.validation import validate_required_columns


def get_available_metrics(ds: EvaluationDataset) -> list[Metric]:
    """
    Get the available metrics for the given dataset.
    E.g. if the dataset contains ("question", "answer", "contexts") columns,
    the available metrics are those that can be evaluated in [qa, qac, qc] mode.
    """
    available_metrics = []
    for metric in ALL_METRICS:
        try:
            validate_required_columns(ds, [metric])
            available_metrics.append(metric)
        except ValueError:
            pass

    return available_metrics

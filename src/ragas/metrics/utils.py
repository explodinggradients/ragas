from ragas.dataset_schema import EvaluationDataset
from ragas.metrics import ALL_METRICS
from ragas.metrics.base import Metric


def validate_required_columns(ds: EvaluationDataset, metric: Metric):
    """
    Checks if the dataset contains all required columns for the given metric.
    """
    required_columns = set(metric.required_columns)
    available_columns = set(ds.features())
    return required_columns.issubset(available_columns)


def get_available_metrics(ds: EvaluationDataset) -> list[Metric]:
    """
    Get the available metrics for the given dataset.
    E.g. if the dataset contains ("question", "answer", "contexts") columns,
    the available metrics are those that can be evaluated in [qa, qac, qc] mode.
    """
    available_metrics = []
    for metric in ALL_METRICS:
        if validate_required_columns(ds, metric):
            available_metrics.append(metric)

    return available_metrics

from __future__ import annotations

import logging

from datasets import Dataset, Sequence

from ragas.dataset_schema import EvaluationDataset
from ragas.metrics.base import Metric

logger = logging.getLogger(__name__)


def remap_column_names(dataset: Dataset, column_map: dict[str, str]) -> Dataset:
    """
    Remap the column names in case dataset uses different column names
    """

    inverse_column_map = {v: k for k, v in column_map.items()}
    return dataset.rename_columns(inverse_column_map)


def handle_deprecated_ground_truths(ds: Dataset) -> Dataset:
    if "ground_truths" in ds.features and "ground_truth" not in ds.features:
        column_names = "ground_truths"
        if (
            isinstance(ds.features[column_names], Sequence)
            and ds.features[column_names].feature.dtype == "string"
        ):
            logger.warning(
                "passing column names as 'ground_truths' is deprecated and will be removed in the next version, please use 'ground_truth' instead. Note that `ground_truth` should be of type string and not Sequence[string] like `ground_truths`"
            )
            gt = [gt[0] for gt in ds["ground_truths"]]
            ds = ds.add_column(
                "ground_truth",
                gt,
                new_fingerprint=ds._fingerprint
                + "a",  # adding random to fingerprint to avoid caching
            )
    return ds


def validate_required_columns(ds: EvaluationDataset, metrics: list[Metric]):
    for m in metrics:
        required_columns = set(m.required_columns)
        available_columns = set(ds.features())
        if not required_columns.issubset(available_columns):
            raise ValueError(
                f"The metric [{m.name}] that that is used requires the following "
                f"additional columns {list(required_columns - available_columns)} "
                f"to be present in the dataset."
            )

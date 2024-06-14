from __future__ import annotations

import logging

from datasets import Dataset, Sequence

from ragas.metrics._context_precision import ContextPrecision
from ragas.metrics.base import EvaluationMode, Metric

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


def validate_column_dtypes(ds: Dataset):
    for column_names in ["question", "answer", "ground_truth"]:
        if column_names in ds.features:
            if ds.features[column_names].dtype != "string":
                raise ValueError(
                    f'Dataset feature "{column_names}" should be of type string'
                )

    for column_names in ["contexts"]:
        if column_names in ds.features:
            if not (
                isinstance(ds.features[column_names], Sequence)
                and ds.features[column_names].feature.dtype == "string"
            ):
                raise ValueError(
                    f'Dataset feature "{column_names}" should be of type'
                    f" Sequence[string], got {type(ds.features[column_names])}"
                )


EVALMODE_TO_COLUMNS = {
    EvaluationMode.qac: ["question", "answer", "contexts"],
    EvaluationMode.qa: ["question", "answer"],
    EvaluationMode.qc: ["question", "contexts"],
    EvaluationMode.gc: ["ground_truth", "contexts"],
    EvaluationMode.ga: ["ground_truth", "answer"],
    EvaluationMode.qga: ["question", "ground_truth", "answer"],
    EvaluationMode.qcg: ["question", "contexts", "ground_truth"],
    EvaluationMode.ca: ["contexts", "summary"],
}


def validate_evaluation_modes(
    ds: Dataset, metrics: list[Metric], evalmode_to_columns: dict = EVALMODE_TO_COLUMNS
):
    """
    validates the dataset and returns the evaluation type

    possible evaluation types
    1. (q,a,c)
    2. (q,a)
    3. (q,c)
    4. (g,a)
    """
    for m in metrics:
        required_columns = set(evalmode_to_columns[m.evaluation_mode])
        available_columns = set(ds.features.keys())
        if not required_columns.issubset(available_columns):
            extra_msg = ""
            if (
                isinstance(m, ContextPrecision)
                and "ground_truth" not in available_columns
            ):
                extra_msg = "Looks like you're trying to use 'context_precision' without ground_truth. Please use consider using  `context_utilization' instead."

            raise ValueError(
                f"The metric [{m.name}] that that is used requires the following "
                f"additional columns {list(required_columns - available_columns)} "
                f"to be present in the dataset. {extra_msg}"
            )

from __future__ import annotations

from datasets import Dataset, Sequence

from ragas.metrics.base import EvaluationMode, Metric


def validate_column_dtypes(ds: Dataset):
    for column_names in ["question", "answer"]:
        if column_names in ds.features:
            if ds.features[column_names].dtype != "string":
                raise ValueError(
                    f'Dataset feature "{column_names}" should be of type string'
                )

    for column_names in ["contexts", "ground_truths"]:
        if column_names in ds.features:
            if not (
                isinstance(ds.features[column_names], Sequence)
                and ds.features[column_names].feature.dtype == "string"
            ):
                raise ValueError(
                    f'Dataset feature "{column_names}" should be of type'
                    " Sequence[string]"
                )


EVALMODE_TO_COLUMNS = {
    EvaluationMode.qac: ["question", "answer", "contexts"],
    EvaluationMode.qa: ["question", "answer"],
    EvaluationMode.qc: ["question", "contexts"],
    EvaluationMode.ga: ["ground_truths", "answer"],
}


def validate_evaluation_modes(ds: Dataset, metrics: list[Metric]):
    """
    validates the dataset and returns the evaluation type

    possible evaluation types
    1. (q,a,c)
    2. (q,a)
    3. (q,c)
    4. (g,a)
    """

    for m in metrics:
        required_columns = set(EVALMODE_TO_COLUMNS[m.evaluation_mode])
        available_columns = set(ds.features.keys())
        if not required_columns.issubset(available_columns):
            raise ValueError(
                f"The metric [{m.name}] that that is used requires the following "
                f"additional columns {list(required_columns - available_columns)} "
                "to be present in the dataset."
            )

from __future__ import annotations

from datasets import Dataset, Sequence

from ragas.metrics.base import EvaluationMode, Metric


def remap_column_names(dataset: Dataset, column_map: dict[str, str]) -> Dataset:
    """
    Remap the column names in case dataset uses different column names
    """
    column_map = {k: v for k, v in column_map.items() if v in dataset.column_names}
    inverse_column_map = {v: k for k, v in column_map.items()}
    return dataset.from_dict(
        {inverse_column_map[name]: dataset[name] for name in column_map.values()}
    )


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
                    f" Sequence[string], got {type(ds.features[column_names])}"
                )


EVALMODE_TO_COLUMNS = {
    EvaluationMode.qac: ["question", "answer", "contexts"],
    EvaluationMode.qa: ["question", "answer"],
    EvaluationMode.qc: ["question", "contexts"],
    EvaluationMode.gc: ["ground_truths", "contexts"],
    EvaluationMode.ga: ["ground_truths", "answer"],
    EvaluationMode.qga: ["question", "ground_truths", "answer"],
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

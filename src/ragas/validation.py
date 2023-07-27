from datasets import Dataset, Sequence

from ragas.metrics.base import EvaluationMode


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
                    f'Dataset feature "{column_names}" should be of type Sequence[string]'
                )


def get_evaluation_mode(ds: Dataset):
    """
    validates the dataset and returns the evaluation type

    possible evaluation types
    1. (q,a,c)
    2. (q,a)
    3. (q,c)
    4. (g,a)
    """
    if (
        "question" in ds.features
        and "answer" in ds.features
        and "contexts" in ds.features
    ):
        return EvaluationMode.qac
    elif "question" in ds.features and "answer" in ds.features:
        return EvaluationMode.qa
    elif "question" in ds.features and "contexts" in ds.features:
        return EvaluationMode.qc
    elif "ground_truths" in ds.features and "answer" in ds.features:
        return EvaluationMode.ga

from __future__ import annotations

import typing as t
from dataclasses import dataclass
from enum import Enum

import numpy as np
from datasets import Dataset, concatenate_datasets
from tqdm import tqdm

from ragas.metrics.base import Metric

EvaluationMode = Enum("EvaluationMode", "generative retrieval grounded")


def get_evaluation_mode(ds: Dataset):
    """
    validates the dataset and returns the evaluation type

    possible evaluation types
    1. (q,a,c)
    2. (q)
    3. (q,c)
    4. (g,a)
    """
    ...


def evaluate(
    metrics: list[Metric],
    dataset: Dataset | None = None,
    questions: list[str] | None = None,
    answers: list[str] | None = None,
    ground_truths: list[str] | None = None,
    contexts: list[list[str]] | None = None,
) -> Result:
    """ """
    if dataset is None:
        data = {}
        if questions is not None:
            data["questions"] = questions
        if answers is not None:
            data["answers"] = answers
        if ground_truths is not None:
            data["ground_truths"] = ground_truths
        if contexts is not None:
            data["contexts"] = contexts

        if len(data) == 0:
            raise ValueError("Provide dataset or (qagc)")
        dataset = Dataset.from_dict(data)

    # TODO: validate EvaluationMode here
    # evaluation_mode = get_evaluation_mode(dataset)

    # TODO: check if all the metrics are compatible with the evaluation mode

    # run the evaluation on dataset with different metrics
    # initialize all the models in the metrics
    [m.init_model() for m in metrics]

    scores = []
    for metric in tqdm(metrics):
        scores.append(metric.score(dataset).select_columns(metric.name))

    return Result(concatenate_datasets(scores))


@dataclass
class Result(dict):
    scores: Dataset

    def __post_init__(self):
        for cn in self.scores.column_names:
            self[cn] = np.mean(self.scores[cn])

    def describe(self):
        description = {}
        for cn in self.scores.column_names:
            description[cn] = {
                "mean": np.mean(self.scores[cn]),
                "25%": np.percentile(self.scores[cn], 25),
                "50%": np.percentile(self.scores[cn], 50),
                "75%": np.percentile(self.scores[cn], 75),
                "min": np.min(self.scores[cn]),
                "max": np.max(self.scores[cn]),
                "std": np.std(self.scores[cn]),
            }
        return description

    def __repr__(self) -> str:
        return super().__repr__()

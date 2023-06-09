from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from datasets import Dataset, concatenate_datasets

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
    dataset: Dataset,
    metrics: list[Metric] | None = None,
) -> Result:
    """
    Run the evaluation on the dataset with different metrics

    Parameters
    ----------
    dataset : Dataset[question: list[str], contexts: list[list[str]], answer: list[str]]
        The dataset in the format of ragas which the metrics will use to score the RAG
        pipeline with

    metrics : list[Metric] , optional
        List of metrics to use for evaluation. If not provided then ragas will run the
        evaluation on the best set of metrics to give a complete view.

    Returns
    -------
    result : Result
        Result object containing the scores of each metric. You can use this do analysis
        later. If the top 3 metrics are provided then it also returns the `ragas_score`
        for the entire pipeline.

    Examples
    --------
    the basic usage is as follows:
    ```
    from ragas import evaluate

    >>> dataset
    Dataset({
        features: ['question', 'ground_truths', 'answer', 'contexts'],
        num_rows: 30
    })

    >>> result = evaluate(dataset)
    >>> print(result["ragas_score"])
    {'ragas_score': 0.860, 'context_relavency': 0.817, 'factuality': 0.892,
    'answer_relevancy': 0.874}
    ```
    """
    if dataset is None:
        raise ValueError("Provide dataset!")

    # TODO: validate EvaluationMode here
    # evaluation_mode = get_evaluation_mode(dataset)

    # TODO: check if all the metrics are compatible with the evaluation mode

    if metrics is None:
        from ragas.metrics import answer_relevancy, context_relevancy, factuality

        metrics = [answer_relevancy, context_relevancy, factuality]

    # run the evaluation on dataset with different metrics
    # initialize all the models in the metrics
    [m.init_model() for m in metrics]

    scores = []
    for metric in metrics:
        scores.append(metric.score(dataset).select_columns(metric.name))

    return Result(scores=concatenate_datasets(scores, axis=1), dataset=dataset)


@dataclass
class Result(dict):
    scores: Dataset
    dataset: Dataset | None = None
    ragas_score: float | None = None

    def __post_init__(self):
        values = []
        for cn in self.scores.column_names:
            value = np.mean(self.scores[cn])
            self[cn] = value
            values.append(value)

        # harmonic mean of all the scores we have
        if len(values) == 3:
            self["ragas_score"] = len(values) / np.sum(1.0 / np.array(values))

    def to_pandas(self, batch_size: int | None = None, batched: bool = False):
        if self.dataset is None:
            raise ValueError("dataset is not provided for the results class")
        assert self.scores.shape[0] == self.dataset.shape[0]
        result_ds = concatenate_datasets([self.dataset, self.scores], axis=1)

        return result_ds.to_pandas(batch_size=batch_size, batched=batched)

    def __repr__(self) -> str:
        scores = self.copy()
        ragas_score = scores.pop("ragas_score")
        score_strs = [f"'ragas_score': {ragas_score:0.4f}"]
        score_strs.extend([f"'{k}': {v:0.4f}" for k, v in scores.items()])
        return "{" + ", ".join(score_strs) + "}"

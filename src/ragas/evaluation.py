from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from datasets import Dataset, concatenate_datasets

from ragas._analytics import EvaluationEvent, track
from ragas.metrics.base import Metric
from ragas.metrics.critique import AspectCritique
from ragas.validation import (
    remap_column_names,
    validate_column_dtypes,
    validate_evaluation_modes,
)


def evaluate(
    dataset: Dataset,
    metrics: list[Metric] | None = None,
    column_map: dict[str, str] = {
        "question": "question",
        "contexts": "contexts",
        "answer": "answer",
        "ground_truths": "ground_truths",
    },
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
    column_map : dict[str, str], optional
        The column names of the dataset to use for evaluation. If the column names of
        the dataset are different from the default ones then you can provide the
        mapping as a dictionary here.

    Returns
    -------
    Result
        Result object containing the scores of each metric. You can use this do analysis
        later. If the top 3 metrics are provided then it also returns the `ragas_score`
        for the entire pipeline.

    Raises
    ------
    ValueError
        if validation fails because the columns required for the metrics are missing or
        if the columns are of the wrong format.

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
    {'ragas_score': 0.860, 'context_precision': 0.817, 'faithfulness': 0.892,
    'answer_relevancy': 0.874}
    ```
    """
    if dataset is None:
        raise ValueError("Provide dataset!")

    if metrics is None:
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        metrics = [answer_relevancy, context_precision, faithfulness, context_recall]

    # remap column names from the dataset
    dataset = remap_column_names(dataset, column_map)

    # validation
    validate_evaluation_modes(dataset, metrics)
    validate_column_dtypes(dataset)

    # run the evaluation on dataset with different metrics
    # initialize all the models in the metrics
    [m.init_model() for m in metrics]

    scores = []
    binary_metrics = []
    for metric in metrics:
        if isinstance(metric, AspectCritique):
            binary_metrics.append(metric.name)
        print(f"evaluating with [{metric.name}]")
        scores.append(metric.score(dataset).select_columns(metric.name))

    # log the evaluation event
    metrics_names = [m.name for m in metrics]
    track(
        EvaluationEvent(
            event_type="evaluation",
            metrics=metrics_names,
            evaluation_mode="",
            num_rows=dataset.shape[0],
        )
    )

    return Result(
        scores=concatenate_datasets(scores, axis=1),
        dataset=dataset,
        binary_columns=binary_metrics,
    )


@dataclass
class Result(dict):
    scores: Dataset
    dataset: Dataset | None = None
    ragas_score: float | None = None
    binary_columns: list[str] = field(default_factory=list)

    def __post_init__(self):
        values = []
        for cn in self.scores.column_names:
            value = np.mean(self.scores[cn])
            self[cn] = value
            if cn not in self.binary_columns:
                values.append(value)

        # harmonic mean of all the scores we have
        if len(values) > 1:
            reciprocal_sum = np.sum(1.0 / np.array(values))  # type: ignore
            self["ragas_score"] = len(values) / reciprocal_sum

    def to_pandas(self, batch_size: int | None = None, batched: bool = False):
        if self.dataset is None:
            raise ValueError("dataset is not provided for the results class")
        assert self.scores.shape[0] == self.dataset.shape[0]
        result_ds = concatenate_datasets([self.dataset, self.scores], axis=1)

        return result_ds.to_pandas(batch_size=batch_size, batched=batched)

    def __repr__(self) -> str:
        scores = self.copy()
        score_strs = []
        if "ragas_score" in scores:
            ragas_score = scores.pop("ragas_score")
            score_strs.append(f"'ragas_score': {ragas_score:0.4f}")
        score_strs.extend([f"'{k}': {v:0.4f}" for k, v in scores.items()])
        return "{" + ", ".join(score_strs) + "}"

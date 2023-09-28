from __future__ import annotations

import typing as t
from dataclasses import dataclass, field

import numpy as np
from datasets import Dataset

from ragas.metrics.answer_similarity import AnswerSimilarity
from ragas.metrics.base import EvaluationMode, MetricWithLLM
from ragas.metrics.faithfulness import Faithfulness

if t.TYPE_CHECKING:
    from langchain.callbacks.manager import CallbackManager


@dataclass
class AnswerCorrectness(MetricWithLLM):
    """
    docs
    """

    name: str = "answer_correctness"
    evaluation_mode: EvaluationMode = EvaluationMode.qga
    batch_size: int = 15
    weights: list[float, float] = field(default_factory=lambda: [0.5, 0.5])
    answer_similarity: AnswerSimilarity | None = None
    faithfulness: Faithfulness | None = None

    def __post_init__(self: t.Self):
        if self.answer_similarity is None:
            self.answer_similarity = AnswerSimilarity()
        if self.faithfulness is None:
            self.faithfulness = Faithfulness()

    def init_model(self: t.Self):
        pass

    def _score_batch(
        self: t.Self,
        dataset: Dataset,
        callbacks: t.Optional[CallbackManager] = None,
        callback_group_name: str = "batch",
    ) -> list[float]:
        if "contexts" in dataset.column_names:
            ds_faithfulness = dataset.remove_columns(["contexts"])
        else:
            ds_faithfulness = dataset

        ds_faithfulness = ds_faithfulness.rename_columns({"ground_truths": "contexts"})
        print(ds_faithfulness.column_names)
        faithfulness_scores = self.faithfulness._score_batch(ds_faithfulness)
        similarity_scores = self.answer_similarity._score_batch(dataset)

        scores = np.vstack([faithfulness_scores, similarity_scores])
        scores = np.average(
            [faithfulness_scores, similarity_scores], axis=0, weights=self.weights
        )

        return scores.tolist()


answer_correctness = AnswerCorrectness()

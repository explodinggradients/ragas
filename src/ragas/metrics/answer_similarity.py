from __future__ import annotations

import typing as t
from dataclasses import dataclass

import numpy as np
from datasets import Dataset
from sentence_transformers import CrossEncoder

from ragas.metrics.base import EvaluationMode, MetricWithLLM

if t.TYPE_CHECKING:
    from langchain.callbacks.manager import CallbackManager


@dataclass
class AnswerSimilarity(MetricWithLLM):
    """
    Scores the semantic similarity of ground truth with generated answer.
    cross encoder score is used to quantify semantic similarity.
    SAS paper: https://arxiv.org/pdf/2108.06130.pdf

    Attributes
    ----------
    name : str
    batch_size : int
        Batch size for openai completion.
    embeddings:
        The cross-encoder model to be used.
        Defaults cross-encoder/stsb-TinyBERT-L-4
        Other good options https://huggingface.co/spaces/mteb/leaderboard
    threshold:
        The threshold if given used to map output to binary
        Default 0.5
    """

    name: str = "answer_similarity"
    evaluation_mode: EvaluationMode = EvaluationMode.ga
    batch_size: int = 15
    embeddings: str | None = None
    threshold: float | None = 0.5

    def __post_init__(self: t.Self):
        if self.embeddings is None:
            self.cross_encoder = CrossEncoder("cross-encoder/stsb-TinyBERT-L-4")

    def init_model(self: t.Self):
        pass

    def _score_batch(
        self: t.Self,
        dataset: Dataset,
        callbacks: t.Optional[CallbackManager] = None,
        callback_group_name: str = "batch",
    ) -> list[float]:
        ground_truths, answers = dataset["ground_truths"], dataset["answer"]
        ground_truths = [item[0] for item in ground_truths]
        inputs = [list(item) for item in list(zip(ground_truths, answers))]
        scores = self.cross_encoder.predict(
            inputs, batch_size=self.batch_size, convert_to_numpy=True
        )

        assert isinstance(scores, np.ndarray), "Expects ndarray"
        if self.threshold:
            scores = scores >= self.threshold  # type: ignore

        return scores.tolist()


answer_similarity = AnswerSimilarity()

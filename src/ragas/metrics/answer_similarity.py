from __future__ import annotations

import typing as t
from dataclasses import dataclass

import numpy as np
from datasets import Dataset
from langchain.embeddings import OpenAIEmbeddings

from ragas.embeddings.embeddings import HuggingfaceEmbeddings
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
        Batch size.
    model_name:
        The model to be used for calculating semantic similarity
        Defaults open-ai-embeddings
        select cross-encoder model for best results
        https://huggingface.co/spaces/mteb/leaderboard
    threshold:
        The threshold if given used to map output to binary
        Default 0.5
    """

    name: str = "answer_similarity"
    evaluation_mode: EvaluationMode = EvaluationMode.ga
    batch_size: int = 15
    model: HuggingfaceEmbeddings | None = None
    threshold: float | None = 0.5

    def __post_init__(self: t.Self):
        if self.model is None:
            self.model = OpenAIEmbeddings()
            self.is_cross_encoder = False
        else:
            self.is_cross_encoder = True if self.model.is_cross_encoder else False
            self.model.encode_kwargs = {"batch_size": self.batch_size}

    def _score_batch(
        self: t.Self,
        dataset: Dataset,
        callbacks: t.Optional[CallbackManager] = None,
        callback_group_name: str = "batch",
    ) -> list[float]:
        ground_truths, answers = dataset["ground_truths"], dataset["answer"]
        ground_truths = [item[0] for item in ground_truths]

        if self.is_cross_encoder:
            inputs = [list(item) for item in list(zip(ground_truths, answers))]
            scores = np.array(self.model.predict(inputs))
        else:
            embeddings_1 = np.array(self.model.embed_documents(ground_truths))
            embeddings_2 = np.array(self.model.embed_documents(answers))
            similarity = embeddings_1 @ embeddings_2.T
            scores = np.diagonal(similarity)

        assert isinstance(scores, np.ndarray), "Expects ndarray"
        if self.threshold:
            scores = scores >= self.threshold  # type: ignore

        return scores.tolist()


answer_similarity = AnswerSimilarity()

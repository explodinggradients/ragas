from __future__ import annotations

import logging
import typing as t
from dataclasses import dataclass

import numpy as np

from ragas.embeddings.base import HuggingfaceEmbeddings
from ragas.metrics.base import EvaluationMode, MetricWithEmbeddings, MetricWithLLM

if t.TYPE_CHECKING:
    from langchain_core.callbacks.base import Callbacks


logger = logging.getLogger(__name__)


@dataclass
class AnswerSimilarity(MetricWithLLM, MetricWithEmbeddings):
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

    name: str = "answer_similarity"  # type: ignore
    evaluation_mode: EvaluationMode = EvaluationMode.ga  # type: ignore
    batch_size: int = 15
    is_cross_encoder: bool = False
    threshold: t.Optional[float] = None

    def __post_init__(self: t.Self):
        # only for cross encoder
        if isinstance(self.embeddings, HuggingfaceEmbeddings):
            self.is_cross_encoder = True if self.embeddings.is_cross_encoder else False
            self.embeddings.encode_kwargs = {
                **self.embeddings.encode_kwargs,
                "batch_size": self.batch_size,
            }

    def init_model(self):
        super().init_model()

    def _score(self, row: t.Dict, callbacks: Callbacks) -> float:
        assert self.embeddings is not None, "embeddings must be set"

        ground_truths, answers = row["ground_truths"], row["answer"]
        ground_truths = [item[0] for item in ground_truths]

        if self.is_cross_encoder and isinstance(self.embeddings, HuggingfaceEmbeddings):
            raise NotImplementedError(
                "async score [ascore()] not implemented for HuggingFace embeddings"
            )
        else:
            embeddings_1 = np.array(self.embeddings.embed_documents(ground_truths))
            embeddings_2 = np.array(self.embeddings.embed_documents(answers))
            similarity = embeddings_1 @ embeddings_2.T
            if similarity.size == 1:
                # If similarity has only one value, directly use this value as scores
                scores = similarity.flatten()
            else:
                # If similarity contains multiple values, extract the diagonal as scores
                scores = np.diagonal(similarity)

        assert isinstance(scores, np.ndarray), "Expects ndarray"
        if self.threshold:
            scores = scores >= self.threshold  # type: ignore

        return scores.tolist()[0]

    async def _ascore(self: t.Self, row: t.Dict, callbacks: Callbacks = []) -> float:
        assert self.embeddings is not None, "embeddings must be set"

        ground_truths, answers = row["ground_truths"], row["answer"]
        ground_truths = [item[0] for item in ground_truths]

        if self.is_cross_encoder and isinstance(self.embeddings, HuggingfaceEmbeddings):
            raise NotImplementedError(
                "async score [ascore()] not implemented for HuggingFace embeddings"
            )
        else:
            embeddings_1 = np.array(
                await self.embeddings.aembed_documents(ground_truths)
            )
            embeddings_2 = np.array(await self.embeddings.aembed_documents(answers))
            similarity = embeddings_1 @ embeddings_2.T
            if similarity.size == 1:
                scores = similarity.flatten()
            else:
                scores = np.diagonal(similarity)

        assert isinstance(scores, np.ndarray), "Expects ndarray"
        if self.threshold:
            scores = scores >= self.threshold  # type: ignore

        return scores.tolist()[0]


answer_similarity = AnswerSimilarity()

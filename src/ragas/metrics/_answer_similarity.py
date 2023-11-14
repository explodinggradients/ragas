from __future__ import annotations

import typing as t
from dataclasses import dataclass, field

import numpy as np
from datasets import Dataset

from ragas.embeddings.base import (
    HuggingfaceEmbeddings,
    OpenAIEmbeddings,
    embedding_factory,
)
from ragas.exceptions import OpenAIKeyNotFound
from ragas.metrics.base import EvaluationMode, MetricWithLLM

if t.TYPE_CHECKING:
    from langchain.callbacks.manager import CallbackManager

    from ragas.embeddings.base import RagasEmbeddings


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
    embeddings: RagasEmbeddings = field(default_factory=embedding_factory)
    is_cross_encoder: bool = False
    threshold: float = 0.5

    def __post_init__(self: t.Self):
        # only for cross encoder
        if isinstance(self.embeddings, HuggingfaceEmbeddings):
            self.is_cross_encoder = True if self.embeddings.is_cross_encoder else False
            self.embeddings.encode_kwargs = {"batch_size": self.batch_size, "convert_to_tensor": True}

    def init_model(self):
        super().init_model()

        if isinstance(self.embeddings, OpenAIEmbeddings):
            if self.embeddings.openai_api_key == "no-key":
                raise OpenAIKeyNotFound

    def _score_batch(
        self: t.Self,
        dataset: Dataset,
        callbacks: t.Optional[CallbackManager] = None,
        callback_group_name: str = "batch",
    ) -> list[float]:
        ground_truths, answers = dataset["ground_truths"], dataset["answer"]
        ground_truths = [item[0] for item in ground_truths]
        self.logs["ground_truths"] += ground_truths

        if self.is_cross_encoder:
            assert isinstance(self.embeddings, HuggingfaceEmbeddings)
            inputs = [list(item) for item in list(zip(ground_truths, answers))]
            scores = np.array(self.embeddings.predict(inputs))
        else:
            embeddings_1 = np.array(self.embeddings.embed_documents(ground_truths))
            embeddings_2 = np.array(self.embeddings.embed_documents(answers))

            # Normalize the embeddings to unit length
            normalized_embeddings_1 = embeddings_1 / np.linalg.norm(embeddings_1, axis=1, keepdims=True)
            normalized_embeddings_2 = embeddings_2 / np.linalg.norm(embeddings_2, axis=1, keepdims=True)

            # Compute the similarity
            similarity = normalized_embeddings_1 @ normalized_embeddings_2.T

            scores = np.diagonal(similarity)
        self.logs["scores"] += scores.tolist()

        assert isinstance(scores, np.ndarray), "Expects ndarray"
        if self.threshold:
            scores = scores >= self.threshold  # type: ignore
            self.logs["thresholded_scores"] += scores.tolist()

        return scores.tolist()


answer_similarity = AnswerSimilarity()

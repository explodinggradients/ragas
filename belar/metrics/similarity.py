from __future__ import annotations

import typing as t
from dataclasses import dataclass

import numpy as np
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer

from belar.metrics.base import Metric

SBERT_METRIC = t.Literal["cosine", "euclidean"]


@dataclass
class SBERTScore(Metric):
    similarity_metric: t.Literal[SBERT_METRIC] = "cosine"
    model_path: str = "all-MiniLM-L6-v2"
    batch_size: int = 1000

    def __post_init__(self):
        self.model = SentenceTransformer(self.model_path)

    @property
    def name(
        self,
    ):
        return f"SBERT_{self.similarity_metric}"

    def is_batchable(self):
        return True

    def score(
        self,
        ground_truth: str | list[str],
        generated_text: str | list[str],
    ):
        if isinstance(ground_truth, str):
            ground_truth = [ground_truth]
        if isinstance(generated_text, str):
            generated_text = [generated_text]

        gndtruth_emb = self.model.encode(
            ground_truth, batch_size=self.batch_size, convert_to_numpy=True
        )
        gentext_emb = self.model.encode(
            generated_text, batch_size=self.batch_size, convert_to_numpy=True
        )

        if self.similarity_metric == "cosine":
            score = np.dot(gndtruth_emb, gentext_emb.T) / (
                norm(gndtruth_emb) * norm(gentext_emb)
            )

        elif self.similarity_metric == "euclidean":
            score = norm(gndtruth_emb - gentext_emb, ord=2)

        else:
            raise ValueError(f"Unkown metrics {self.similarity_metric}")

        return score


__all__ = ["SBERTScore"]

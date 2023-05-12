from __future__ import annotations

import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass

from datasets import Dataset


@dataclass
class Metric(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def is_batchable(self) -> bool:
        ...

    @abstractmethod
    def score(self, ground_truth, generated_text) -> float | list[float]:
        ...

    def __call__(self, row):
        score = self.score(row["ground_truth"], row["generated_text"])
        row[f"{self.name}_score"] = score

        return row


@dataclass
class Evaluation:
    metrics: list[Metric]

    def eval(
        self, ground_truth: Dataset, generated_text: t.Sequence, batched: bool = False
    ):
        ds = ground_truth.add_column("generated_text", generated_text)
        scores_list = []
        for metric in self.metrics:
            scores = ds.map(metric, batched=batched)[f"{metric.name}_score"]
            scores_list.append(scores)

        return scores_list

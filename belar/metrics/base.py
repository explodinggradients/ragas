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


@dataclass
class Evaluation:
    metrics: list[Metric]
    batched: bool = False

    def eval(self, ground_truth: list[list[str]], generated_text: list[list[str]]):
        ds = ground_truth.add_column("generated_text", generated_text)
        ds = ds.map(self._get_score, batched=self.batched)

        return ds

    def _get_score(self, row):
        for metric in self.metrics:
            score = metric.score(row["ground_truth"], row["generated_text"])
            row[f"{metric.name}_score"] = score

        return row

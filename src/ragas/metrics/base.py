"""
Q - question
A - answer: generated_text from RAG pipeline
C - contexts: context used for generation
G - ground_truths: ground truth answer
"""
from __future__ import annotations

import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import floor

from datasets import Dataset


@dataclass
class Metric(ABC):
    @property
    @abstractmethod
    def batch_size(self: t.Self) -> int:
        ...

    @property
    @abstractmethod
    def name(self: t.Self) -> str:
        """
        the metric name
        """
        ...

    @abstractmethod
    def init_model():
        """
        This method will lazy initialize the model.
        """
        ...

    @abstractmethod
    def score(self: t.Self, dataset: Dataset) -> Dataset:
        ...

    def get_batches(self, dataset_size: int):
        tail = dataset_size % self.batch_size
        num_batches = floor(dataset_size / self.batch_size)
        batches = [
            range(i, i + self.batch_size)
            for i in range(0, self.batch_size * num_batches, self.batch_size)
        ]
        if tail != 0:
            batches.append(
                range(
                    self.batch_size * num_batches, self.batch_size * num_batches + tail
                )
            )

        return batches

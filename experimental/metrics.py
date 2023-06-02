from __future__ import annotations

import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class GenerationMetric(ABC):
    @property
    @abstractmethod
    def name(self: t.Self) -> str:
        """
        the metric name
        """
        ...

    @property
    @abstractmethod
    def is_batchable(self: t.Self) -> bool:
        """
        Attribute to check if this metric is is_batchable
        """
        ...

    @abstractmethod
    def init_model():
        """
        This method will lazy initialize the model.
        """
        ...

    @abstractmethod
    def score(
        self: t.Self, questions: list[str], context: list[list[str]], answer: list[str]
    ):
        """
        Return the NLI score for each (q, c, a) pair
        """
        ...

"""
Q - questions
A - answers: generated_text from RAG pipeline
C - contexts: context used for generation
G - ground_truths: ground truth answer
"""
from __future__ import annotations

import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Metric(ABC):
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
        self: t.Self, ground_truth: list[str], generated_text: list[str]
    ) -> list[float]:
        """
        Run the metric on the ground_truth and generated_text and return score.
        """
        ...


@dataclass
class GAmetric(Metric, ABC):
    @abstractmethod
    def score(self: t.Self, ground_truth: list[str], answer: list[str]) -> list[float]:
        """
        Run the metric on the ground_truth and generated_text and return score.
        """
        ...


@dataclass
class QCAMetric(Metric, ABC):
    @abstractmethod
    def score(
        self: t.Self, questions: list[str], context: list[list[str]], answer: list[str]
    ):
        """
        Return the NLI score for each (q, c, a) pair
        """
        ...


@dataclass
class QAMetric(Metric, ABC):
    @abstractmethod
    def score(self: t.Self, questions: list[str], answer: list[str]):
        """
        Return the NLI score for each (q, c, a) pair
        """
        ...

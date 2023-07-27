"""
Q - question
A - answer: generated_text from RAG pipeline
C - contexts: context used for generation
G - ground_truths: ground truth answer
"""
from __future__ import annotations

import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from math import floor

from datasets import Dataset
from langchain.chat_models import ChatOpenAI
from langchain.chat_models.base import BaseChatModel
from langchain.llms.base import BaseLLM


def make_batches(total_size: int, batch_size: int) -> list[range]:
    """
    Take a total size and batch size and return a list of ranges for the batches
    """
    tail = total_size % batch_size
    num_batches = floor(total_size / batch_size)
    batches = [
        range(i, i + batch_size) for i in range(0, batch_size * num_batches, batch_size)
    ]
    if tail != 0:
        batches.append(range(batch_size * num_batches, batch_size * num_batches + tail))

    return batches


EvaluationMode = Enum("EvaluationMode", "qac qa qc ga")


@dataclass
class Metric(ABC):
    batch_size: int

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def evaluation_mode(self) -> EvaluationMode:
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

    def get_batches(self, dataset_size: int) -> list[range]:
        return make_batches(dataset_size, self.batch_size)


def _llm_factory():
    return ChatOpenAI(model_name="gpt-3.5-turbo-16k")  # type: ignore


@dataclass
class MetricWithLLM(Metric):
    llm: BaseLLM | BaseChatModel = field(default_factory=_llm_factory)

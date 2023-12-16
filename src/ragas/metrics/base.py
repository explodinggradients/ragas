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
from langchain.callbacks.manager import CallbackManager, trace_as_chain_group
from tqdm import tqdm

from ragas.embeddings.base import RagasEmbeddings
from ragas.llms import llm_factory

if t.TYPE_CHECKING:
    from langchain.callbacks.base import Callbacks

    from ragas.llms import BaseRagasLLM


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


EvaluationMode = Enum("EvaluationMode", "qac qa qc gc ga qga qcg")


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
    def init_model(self):
        """
        This method will lazy initialize the model.
        """
        ...

    def score(
        self: t.Self,
        dataset: Dataset,
        callbacks: t.Optional[Callbacks] = None,
    ) -> float:
        raise NotImplemented

    @abstractmethod
    def ascore(
        self: t.Self, dataset: Dataset, callbacks: t.Optional[Callbacks] = None
    ) -> float:
        ...


@dataclass
class MetricWithLLM(Metric):
    llm: t.Optional[BaseRagasLLM] = None

    def init_model(self):
        """
        Init any models in the metric, this is invoked before evaluate()
        to load all the models
        Also check if the api key is valid for OpenAI and AzureOpenAI
        """
        if self.llm is None:
            raise ValueError(
                f"Metric '{self.name}' has no valid LLM provided. Please initantiate a the metric with an LLM to run."  # noqa
            )
        if hasattr(self.llm, "validate_api_key"):
            self.llm.validate_api_key()
        if hasattr(self, "embeddings"):
            # since we are using Langchain Embeddings directly, we need to check this
            if hasattr(self.embeddings, "validate_api_key"):
                self.embeddings = t.cast(RagasEmbeddings, self.embeddings)
                self.embeddings.validate_api_key()

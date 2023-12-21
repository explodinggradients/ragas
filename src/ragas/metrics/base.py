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
from langchain_core.callbacks import CallbackManager, CallbackManagerForChainGroup
from tqdm import tqdm

from ragas.callbacks import new_group
from ragas.embeddings.base import BaseRagasEmbeddings
from ragas.llms import llm_factory

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks

    from ragas.llms import BaseRagasLLM


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
        row: t.Dict,
        callbacks: Callbacks = [],
    ) -> float:
        rm, group_cm = new_group(
            self.name, inputs=row, callbacks=callbacks, is_async=False
        )
        try:
            score = self._score(row=row, callbacks=group_cm)
        except Exception as e:
            if not group_cm.ended:
                rm.on_chain_error(e)
            raise e
        else:
            if not group_cm.ended:
                rm.on_chain_end({"output": score})
        return score

    # @abstractmethod
    def _score(self, row: t.Dict, callbacks: Callbacks = []) -> float:
        ...

    async def ascore(self: t.Self, row: t.Dict, callbacks: Callbacks = []) -> float:
        rm, group_cm = new_group(
            self.name, inputs=row, callbacks=callbacks, is_async=True
        )
        try:
            score = await self._ascore(row=row, callbacks=group_cm)
        except Exception as e:
            if not group_cm.ended:
                rm.on_chain_error(e)
            raise e
        else:
            if not group_cm.ended:
                rm.on_chain_end({"output": score})
        return score

    @abstractmethod
    async def _ascore(self, row: t.Dict, callbacks: Callbacks = []) -> float:
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
                f"Metric '{self.name}' has no valid LLM provided (self.llm is None). Please initantiate a the metric with an LLM to run."  # noqa
            )
        if hasattr(self.llm, "validate_api_key"):
            self.llm.validate_api_key()
        if hasattr(self, "embeddings"):
            # since we are using Langchain Embeddings directly, we need to check this
            if hasattr(self.embeddings, "validate_api_key"):
                self.embeddings = t.cast(BaseRagasEmbeddings, self.embeddings)
                self.embeddings.validate_api_key()

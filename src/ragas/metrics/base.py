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

from ragas.embeddings.base import RagasEmbeddings
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
        data_row: t.Dict,
        callbacks: t.Optional[Callbacks] = None,
    ) -> float:
        raise NotImplemented

    async def ascore(
        self: t.Self, data_row: t.Dict, callbacks: Callbacks = []
    ) -> float:
        if isinstance(callbacks, list):
            cm = CallbackManager.configure(inheritable_callbacks=callbacks)
        else:
            cm = t.cast(CallbackManager, callbacks)

        rm = cm.on_chain_start({"name": self.name}, data_row)
        child_cm = rm.get_child()
        group_cm = CallbackManagerForChainGroup(
            child_cm.handlers,
            child_cm.inheritable_handlers,
            child_cm.parent_run_id,
            parent_run_manager=rm,
            tags=child_cm.tags,
            inheritable_tags=child_cm.inheritable_tags,
            metadata=child_cm.metadata,
            inheritable_metadata=child_cm.inheritable_metadata,
        )
        try:
            score = await self._ascore(data_row=data_row, callbacks=group_cm)
        except Exception as e:
            if not group_cm.ended:
                rm.on_chain_error(e)
            raise e
        else:
            if not group_cm.ended:
                rm.on_chain_end({"output": score})
        return score

    @abstractmethod
    async def _ascore(self, data_row: t.Dict, callbacks: Callbacks = []) -> float:
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
                self.embeddings = t.cast(RagasEmbeddings, self.embeddings)
                self.embeddings.validate_api_key()

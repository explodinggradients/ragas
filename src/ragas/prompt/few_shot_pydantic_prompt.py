from __future__ import annotations

import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel

from ragas.llms.base import BaseRagasLLM
from ragas.prompt.pydantic_prompt import PydanticPrompt

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks

    from ragas.llms.base import BaseRagasLLM

# type variables for input and output models
InputModel = t.TypeVar("InputModel", bound=BaseModel)
OutputModel = t.TypeVar("OutputModel", bound=BaseModel)


class ExampleStore(ABC):
    @abstractmethod
    def get_examples(self, data: BaseModel, top_k: int = 5) -> t.List[BaseModel]:
        pass

    @abstractmethod
    def add_example(self, input: BaseModel, output: BaseModel):
        pass


@dataclass
class InMemoryExampleStore(ExampleStore):
    embedding_fn: t.Callable[[BaseModel], t.List[float]]
    examples: t.List[t.Tuple[BaseModel, BaseModel]] = field(default_factory=list)
    embeddings: t.List[t.List[float]] = field(default_factory=list)

    def add_example(self, input: BaseModel, output: BaseModel):
        pass

    def get_examples(self, data: BaseModel, top_k: int = 5) -> t.List[BaseModel]:
        pass

    def distance(self, a: t.List[float], b: t.List[float]) -> float:
        pass


class FewShotPydanticPrompt(PydanticPrompt, t.Generic[InputModel, OutputModel]):
    async def generate_multiple(
        self,
        llm: BaseRagasLLM,
        data: InputModel,
        n: int = 1,
        temperature: t.Optional[float] = None,
        stop: t.Optional[t.List[str]] = None,
        callbacks: t.Optional[Callbacks] = None,
        retries_left: int = 3,
    ) -> t.List[OutputModel]:
        self.examples = self.examples[:n]
        return await super().generate_multiple(
            llm, data, n, temperature, stop, callbacks, retries_left
        )

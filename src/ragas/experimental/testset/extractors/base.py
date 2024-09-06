import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass

from ragas_experimental.llms.prompt import PydanticPrompt

from ragas.experimental.testset.graph import Node
from ragas.llms.base import BaseRagasLLM


@dataclass
class BaseExtractor(ABC):
    name: str

    def extract(self, node: Node) -> t.Tuple[str, t.Any]:
        return self._extract(node)

    @abstractmethod
    def _extract(self, node: Node) -> t.Tuple[str, t.Any]:
        pass


@dataclass
class LLMBasedExtractor(BaseExtractor):
    llm: BaseRagasLLM
    prompt: PydanticPrompt

import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ragas.experimental.testset.graph import Node
from ragas.experimental.testset.graph_transforms import Extractor
from ragas.llms.base import BaseRagasLLM, llm_factory


@dataclass
class BaseExtractor(ABC):
    property_name: str

    async def extract(self, node: Node) -> t.Tuple[str, t.Any]:
        return await self._extract(node)

    @abstractmethod
    async def _extract(self, node: Node) -> t.Tuple[str, t.Any]:
        pass


@dataclass
class LLMBasedExtractor(Extractor):
    llm: BaseRagasLLM = field(default_factory=llm_factory)
    merge_if_possible: bool = True

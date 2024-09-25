import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

from pydantic import BaseModel

from ragas.experimental.testset.graph import KnowledgeGraph, Node
from ragas.llms import BaseRagasLLM, llm_factory


class UserInputLength(str, Enum):
    LONG = "long"
    MEDIUM = "medium"
    SHORT = "short"


class UserInputStyle(str, Enum):
    MISSPELLED = "Misspelled queries"
    PERFECT_GRAMMAR = "Perfect grammar"
    POOR_GRAMMAR = "Poor grammar"
    WEB_SEARCH_LIKE = "Web search like queries"


class BasicDistribution(BaseModel):
    nodes: t.List[Node]
    style: UserInputStyle
    length: UserInputLength


Distribution = t.TypeVar("Distribution", bound=BasicDistribution)


@dataclass
class BaseTestsetGenerator(ABC, t.Generic[Distribution]):
    llm: BaseRagasLLM = field(default_factory=llm_factory)

    @abstractmethod
    async def generate_user_input(
        self,
        distribution: Distribution,
    ) -> str:
        pass

    @abstractmethod
    async def generate_reference(self, question: str, chunks: t.List[Node]) -> str:
        pass

    @abstractmethod
    async def critic_user_input(self, question: str) -> bool:
        pass

    @abstractmethod
    async def modify_user_input(self, question: str, distribution: Distribution) -> str:
        pass

    @abstractmethod
    async def generate_distributions(
        self, n: int, knowledge_graph: KnowledgeGraph
    ) -> t.List[Distribution]:
        pass

    @staticmethod
    def make_source_text(distribution: Distribution) -> str:
        page_contents = []
        for node in distribution.nodes:
            page_contents.append(node.get_property("page_content"))
        return "\n\n".join(page_contents)

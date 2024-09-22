import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

from pydantic import BaseModel

from ragas.executor import Executor
from ragas.experimental.testset.graph import KnowledgeGraph, Node
from ragas.llms import BaseRagasLLM, llm_factory


class QuestionLength(str, Enum):
    LONG = "long"
    MEDIUM = "medium"
    SHORT = "short"


class QuestionStyle(str, Enum):
    MISSPELLED = "Misspelled queries"
    PERFECT_GRAMMAR = "Perfect grammar"
    POOR_GRAMMAR = "Poor grammar"
    WEB_SEARCH_LIKE = "Web search like queries"


class BasicDistribution(BaseModel):
    nodes: t.List[Node]
    style: QuestionStyle
    length: QuestionLength


@dataclass
class BaseTestsetGenerator(ABC):
    llm: BaseRagasLLM = field(default_factory=llm_factory)

    @abstractmethod
    async def generate_question(
        self,
        distribution: BasicDistribution,
    ) -> str:
        pass

    @abstractmethod
    async def generate_answer(self, question: str, chunks: t.List[Node]) -> str:
        pass

    @abstractmethod
    async def critic_question(self, question: str) -> bool:
        pass

    @abstractmethod
    async def generate_distributions(
        self, n: int, knowledge_graph: KnowledgeGraph
    ) -> t.List[BasicDistribution]:
        pass

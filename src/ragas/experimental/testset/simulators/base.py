from __future__ import annotations

import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

from pydantic import BaseModel

from ragas.experimental.prompt import PromptMixin
from ragas.experimental.testset.graph import KnowledgeGraph, Node
from ragas.llms import BaseRagasLLM, llm_factory

if t.TYPE_CHECKING:
    from ragas.dataset_schema import BaseEvalSample


class UserInputLength(str, Enum):
    LONG = "long"
    MEDIUM = "medium"
    SHORT = "short"


class UserInputStyle(str, Enum):
    MISSPELLED = "Misspelled queries"
    PERFECT_GRAMMAR = "Perfect grammar"
    POOR_GRAMMAR = "Poor grammar"
    WEB_SEARCH_LIKE = "Web search like queries"


class BaseScenario(BaseModel):
    nodes: t.List[Node]
    style: UserInputStyle
    length: UserInputLength


Scenario = t.TypeVar("Scenario", bound=BaseScenario)


@dataclass
class BaseSimulator(ABC, t.Generic[Scenario], PromptMixin):
    name: str = ""
    llm: BaseRagasLLM = field(default_factory=llm_factory)

    def __post_init__(self):
        if not self.name:
            self.name = self.__class__.__name__

    @abstractmethod
    async def generate_scenarios(
        self, n: int, knowledge_graph: KnowledgeGraph
    ) -> t.List[Scenario]:
        pass

    @abstractmethod
    async def generate_sample(self, scenario: Scenario) -> BaseEvalSample:
        pass

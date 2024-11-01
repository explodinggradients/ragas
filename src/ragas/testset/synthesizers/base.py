from __future__ import annotations

import logging
import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

from pydantic import BaseModel

from ragas.callbacks import new_group
from ragas.llms import BaseRagasLLM, llm_factory
from ragas.prompt import PromptMixin
from ragas.prompt.pydantic_prompt import PydanticPrompt
from ragas.testset.graph import KnowledgeGraph, Node
from ragas.testset.synthesizers.prompts import (
    NodeSummaries,
    Persona,
    PersonaGenerationPrompt,
    PersonasList,
)

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks

    from ragas.dataset_schema import BaseSample

import random

logger = logging.getLogger(__name__)


def default_filter(node: Node) -> bool:

    if node.type.name == "DOCUMENT" and node.properties.get("summary") is not None:
        return True
    else:
        return random.random() < 0.1


class QueryLength(str, Enum):
    """
    Enumeration of query lengths. Available options are: LONG, MEDIUM, SHORT
    """

    LONG = "long"
    MEDIUM = "medium"
    SHORT = "short"


class QueryStyle(str, Enum):
    """
    Enumeration of query styles. Available options are: MISSPELLED, PERFECT_GRAMMAR, POOR_GRAMMAR, WEB_SEARCH_LIKE
    """

    MISSPELLED = "Misspelled queries"
    PERFECT_GRAMMAR = "Perfect grammar"
    POOR_GRAMMAR = "Poor grammar"
    WEB_SEARCH_LIKE = "Web search like queries"


@dataclass
class PersonaGenerator:

    llm: BaseRagasLLM
    prompt: PydanticPrompt = PersonaGenerationPrompt()
    filter_nodes: t.Callable[[Node], bool] = field(
        default_factory=lambda: default_filter
    )
    max_tokens: int = 4000

    async def generate_from_kg(self, kg: KnowledgeGraph) -> PersonasList:

        texts = []
        nodes = [node for node in kg.nodes if self.filter_nodes(node)]
        for node in nodes:
            text = node.properties.get("summary") or node.properties.get(
                "topic_description"
            )
            if text is None:
                logger.warning(
                    f"Node {node} does not have a summary or topic description."
                )
            texts.append(text)

        random.shuffle(texts)
        prompt_input = NodeSummaries(summaries=texts[: self.max_tokens])
        response = await self.prompt.generate(data=prompt_input, llm=self.llm)
        return response


class BaseScenario(BaseModel):
    """
    Base class for representing a scenario for generating test samples.

    Attributes
    ----------
    nodes : List[Node]
        List of nodes involved in the scenario.
    style : QueryStyle
        The style of the query.
    length : QueryLength
        The length of the query.
    """

    nodes: t.List[Node]
    style: QueryStyle
    length: QueryLength
    persona: Persona


Scenario = t.TypeVar("Scenario", bound=BaseScenario)


@dataclass
class BaseSynthesizer(ABC, t.Generic[Scenario], PromptMixin):
    """
    Base class for synthesizing scenarios and samples.
    """

    name: str = ""
    llm: BaseRagasLLM = field(default_factory=llm_factory)

    def __post_init__(self):
        if not self.name:
            self.name = self.__class__.__name__

    async def generate_scenarios(
        self,
        n: int,
        knowledge_graph: KnowledgeGraph,
        callbacks: t.Optional[Callbacks] = None,
        persona_list: t.Optional[PersonasList] = None,
    ) -> t.List[Scenario]:
        callbacks = callbacks or []
        scenario_generation_rm, scenario_generation_group = new_group(
            name=self.name,
            inputs={"n": n, "knowledge_graph": str(knowledge_graph)},
            callbacks=callbacks,
        )
        scenarios = await self._generate_scenarios(
            n, knowledge_graph, scenario_generation_group, persona_list
        )
        scenario_generation_rm.on_chain_end(outputs={"scenarios": scenarios})
        return scenarios

    @abstractmethod
    async def _generate_scenarios(
        self,
        n: int,
        knowledge_graph: KnowledgeGraph,
        callbacks: Callbacks,
        persona_list: t.Optional[PersonasList] = None,
    ) -> t.List[Scenario]:
        pass

    async def generate_sample(
        self, scenario: Scenario, callbacks: t.Optional[Callbacks] = None
    ) -> BaseSample:
        callbacks = callbacks or []

        # new group for Sample Generation
        sample_generation_rm, sample_generation_grp = new_group(
            name=self.name,
            inputs={"scenario": scenario},
            callbacks=callbacks,
        )
        sample = await self._generate_sample(scenario, sample_generation_grp)
        sample_generation_rm.on_chain_end(outputs={"sample": sample})

        return sample

    @abstractmethod
    async def _generate_sample(
        self, scenario: Scenario, callbacks: Callbacks
    ) -> BaseSample:
        pass

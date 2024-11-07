from __future__ import annotations

import logging
import typing as t
from dataclasses import dataclass

import numpy as np

from ragas.prompt import PydanticPrompt
from ragas.testset.graph import KnowledgeGraph
from ragas.testset.persona import Persona, PersonaList
from ragas.testset.synthesizers.base import BaseScenario
from ragas.testset.synthesizers.prompts import (
    ThemesPersonasInput,
    ThemesPersonasMatchingPrompt,
)

from .base import SingleHopQuerySynthesizer

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks

logger = logging.getLogger(__name__)


class SingleHopScenario(BaseScenario):
    """
    Scenario for multi-hop queries.

    Attributes
    ----------
    term: str
        The theme of the query.
    """

    term: str


@dataclass
class SingleHopSpecificQuerySynthesizer(SingleHopQuerySynthesizer):

    name: str = "single_hop_specifc_query_synthesizer"
    theme_persona_matching_prompt: PydanticPrompt = ThemesPersonasMatchingPrompt()

    async def _generate_scenarios(
        self,
        n: int,
        knowledge_graph: KnowledgeGraph,
        persona_list: t.List[Persona],
        callbacks: Callbacks,
    ) -> t.List[SingleHopScenario]:
        """
        Generates a list of scenarios on type SingleHopSpecificQuerySynthesizer
        Steps to generate scenarios:
        1. Find nodes with CHUNK type and entities property
        2. Calculate the number of samples that should be created per node to get n samples in total
        3. For each node
            a. Find the entities associated with the node
            b. Map personas to the entities to create query
            c. Prepare all possible combinations of (node, entities, personas, style, length) as base scenarios
            d. Sample num_sample_per_node (step 2) scenarios from base scenarios
        4. Return the list of scenarios
        """

        property_name = "entities"
        nodes = []
        for node in knowledge_graph.nodes:
            if (
                node.type.name == "CHUNK"
                and node.get_property(property_name) is not None
            ):
                nodes.append(node)

        samples_per_node = int(np.ceil(n / len(nodes)))

        scenarios = []
        for node in nodes:
            if len(scenarios) >= n:
                break
            themes = node.get_property(property_name)
            prompt_input = ThemesPersonasInput(themes=themes, personas=persona_list)
            persona_concepts = await self.theme_persona_matching_prompt.generate(
                data=prompt_input, llm=self.llm, callbacks=callbacks
            )
            base_scenarios = self.prepare_combinations(
                node, themes, PersonaList(personas=persona_list), persona_concepts
            )
            scenarios.extend(self.sample_combinations(base_scenarios, samples_per_node))

        return scenarios

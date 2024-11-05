from __future__ import annotations

import logging
import typing as t
from dataclasses import dataclass

import numpy as np

from ragas.prompt import PydanticPrompt
from ragas.testset.graph import KnowledgeGraph
from ragas.testset.graph_queries import get_child_nodes
from ragas.testset.persona import PersonasList
from ragas.testset.synthesizers.multi_hop.base import (
    MultiHopQuerySynthesizer,
    MultiHopScenario,
)
from ragas.testset.synthesizers.multi_hop.prompts import (
    ConceptCombinationPrompt,
    ConceptsList,
)
from ragas.testset.synthesizers.prompts import (
    ThemesList,
    ThemesPersonasInput,
    ThemesPersonasMatchingPrompt,
)

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks

logger = logging.getLogger(__name__)


@dataclass
class MultiHopAbstractQuerySynthesizer(MultiHopQuerySynthesizer):
    """
    Synthesizes abstract multi-hop queries from given knowledge graph.

    Attributes
    ----------
    """

    concept_combination_prompt: PydanticPrompt = ConceptCombinationPrompt()
    theme_persona_matching_prompt: PydanticPrompt = ThemesPersonasMatchingPrompt()

    async def _generate_scenarios(
        self,
        n: int,
        knowledge_graph: KnowledgeGraph,
        persona_list: PersonasList,
        callbacks: Callbacks,
    ) -> t.List[MultiHopScenario]:

        node_clusters = knowledge_graph.find_indirect_clusters(
            relationship_condition=lambda rel: (
                True if rel.get_property("summary_similarity") else False
            ),
            depth_limit=3,
        )
        logger.info("found %d clusters", len(node_clusters))
        scenarios = []

        num_sample_per_cluster = int(np.ceil(n / len(node_clusters)))

        for cluster in node_clusters:
            if len(scenarios) >= n:
                break
            nodes = []
            for node in cluster:
                child_nodes = get_child_nodes(node, knowledge_graph, level=1)
                if child_nodes:
                    nodes.extend(child_nodes)
                else:
                    nodes.append(node)

            base_scenarios = []
            node_themes = [node.properties.get("themes", []) for node in nodes]
            prompt_input = ConceptsList(
                lists_of_concepts=node_themes, max_combinations=num_sample_per_cluster
            )
            concept_combination = await self.concept_combination_prompt.generate(
                data=prompt_input, llm=self.llm, callbacks=callbacks
            )
            flattened_themes = [
                theme
                for sublist in concept_combination.combinations
                for theme in sublist
            ]
            themes_list = ThemesList(themes=flattened_themes)
            prompt_input = ThemesPersonasInput(
                themes=themes_list, personas=persona_list
            )
            persona_concepts = await self.theme_persona_matching_prompt.generate(
                data=prompt_input, llm=self.llm, callbacks=callbacks
            )

            base_scenarios = self.prepare_combinations(
                nodes,
                concept_combination.combinations,
                persona_list,
                persona_concepts,
                property_name="themes",
            )
            base_scenarios = self.sample_diverse_combinations(
                base_scenarios, num_sample_per_cluster
            )
            scenarios.extend(base_scenarios)

        return scenarios

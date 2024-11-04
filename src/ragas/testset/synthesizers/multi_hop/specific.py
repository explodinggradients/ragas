from __future__ import annotations

import logging
import typing as t
from dataclasses import dataclass

import numpy as np

from ragas.prompt import PydanticPrompt
from ragas.testset.graph import KnowledgeGraph
from ragas.testset.persona import PersonasList
from ragas.testset.synthesizers.multi_hop.base import (
    MultiHopQuerySynthesizer,
    MultiHopScenario,
)
from ragas.testset.synthesizers.multi_hop.prompts import QueryAnswerGenerationPrompt
from ragas.testset.synthesizers.prompts import (
    ThemesList,
    ThemesPersonasInput,
    ThemesPersonasMatchingPrompt,
)

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks

logger = logging.getLogger(__name__)


@dataclass
class MultiHopSpecificQuerySynthesizer(MultiHopQuerySynthesizer):
    """
    Synthesizes overlap based queries by choosing specific chunks and generating a
    keyphrase from them and then generating queries based on that.

    Attributes
    ----------
    generate_query_prompt : PydanticPrompt
        The prompt used for generating the query.
    """

    theme_persona_matching_prompt: PydanticPrompt = ThemesPersonasMatchingPrompt()
    generate_query_reference_prompt: PydanticPrompt = QueryAnswerGenerationPrompt()

    async def _generate_scenarios(
        self,
        n: int,
        knowledge_graph: KnowledgeGraph,
        persona_list: PersonasList,
        callbacks: Callbacks,
    ) -> t.List[MultiHopScenario]:

        cluster_dict = knowledge_graph.find_direct_clusters(
            relationship_condition=lambda rel: (
                True if rel.type == "entities_overlap" else False
            )
        )

        valid_relationships = [
            rel
            for rel in knowledge_graph.relationships
            if rel.type == "entities_overlap"
        ]

        node_clusters = []
        for key_node, list_of_nodes in cluster_dict.items():
            for node in list_of_nodes:
                node_clusters.append((key_node, node))

        logger.info("found %d clusters", len(cluster_dict))
        scenarios = []
        num_sample_per_cluster = int(np.ceil(n / len(node_clusters)))

        for cluster in node_clusters:
            if len(scenarios) < n:
                key_node, node = cluster
                overlapped_items = []
                for rel in valid_relationships:
                    if rel.source == key_node and rel.target == node:
                        overlapped_items = rel.get_property("overlapped_items")
                        break
                print(overlapped_items)
                if overlapped_items:
                    themes = ThemesList(themes=list(dict(overlapped_items).keys()))
                    prompt_input = ThemesPersonasInput(
                        themes=themes, personas=persona_list
                    )
                    persona_concepts = (
                        await self.theme_persona_matching_prompt.generate(
                            data=prompt_input, llm=self.llm, callbacks=callbacks
                        )
                    )
                    overlapped_items = [list(item) for item in overlapped_items]
                    base_scenarios = self.prepare_combinations(
                        [key_node, node],
                        overlapped_items,
                        persona_concepts,
                        property_name="entities",
                    )
                    base_scenarios = self.sample_diverse_combinations(
                        base_scenarios, num_sample_per_cluster
                    )
                    scenarios.extend(base_scenarios)

        scenarios_to_return = []
        for scenario in scenarios:
            if len(scenarios_to_return) < n:
                scenarios_to_return.append(
                    MultiHopScenario(
                        combinations=scenario["combination"],
                        nodes=scenario["nodes"],
                        style=scenario["style"],
                        length=scenario["length"],
                        persona=persona_list[scenario["persona"]],
                    )
                )

        return scenarios_to_return

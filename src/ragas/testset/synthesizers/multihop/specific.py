from __future__ import annotations

import logging
import random
import typing as t
from dataclasses import dataclass


import numpy as np
import logging

from ragas import SingleTurnSample
from ragas.prompt import PydanticPrompt
from ragas.testset.graph import KnowledgeGraph
from ragas.testset.synthesizers.base import BaseScenario, QueryLength, QueryStyle
from ragas.testset.synthesizers.base_query import QuerySynthesizer
from ragas.testset.persona import PersonasList
from ragas.testset.synthesizers.prompts import (ThemesPersonasInput, ThemesPersonasMatchingPrompt, ThemesList, QueryAnswerGenerationPrompt, QueryConditions)
from ragas.testset.synthesizers.multihop.sampling import sample_diverse_combinations

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks

logger = logging.getLogger(__name__)


class OverlapBasedScenario(BaseScenario):
    """
    Represents a scenario for generating overlap based queries.
    Also inherits attributes from [BaseScenario][ragas.testset.synthesizers.base.BaseScenario].

    Attributes
    ----------
    keyphrase : str
        The keyphrase of the overlap based scenario.
    """

    keyphrases: t.List[str]
    
    
    
@dataclass
class MultiHopOverlapBasedSynthesizer(QuerySynthesizer):
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

    def _prepare_seed_node_persona_mapping(self, nodes, combinations: t.List[t.List[str]], persona_concepts) -> t.Dict[str, t.List[str]]:
        
        possible_combinations = []
        for combination in combinations:
            dict = {"combination": combination}
            valid_personas = []
            for persona, concept_list in persona_concepts.mapping.items():
                concept_list = [c.lower() for c in concept_list]
                if any(concept.lower() in concept_list for concept in combination):
                    valid_personas.append(persona)
            dict["personas"] = valid_personas
            valid_nodes = []
            for node in nodes:
                node_themes = [theme.lower() for theme in node.get_property("entities")]
                if node.get_property("entities") and any(
                    concept.lower() in node_themes for concept in combination
                ):
                    valid_nodes.append(node)
                    
            dict["nodes"] = valid_nodes
            dict["styles"] = list(QueryStyle)
            dict["lengths"] = list(QueryLength)
            
            possible_combinations.append(dict)
        return possible_combinations

    
    async def _generate_scenarios(
        self,
        n: int,
        knowledge_graph: KnowledgeGraph,
        callbacks: Callbacks,
        persona_list: t.Optional[PersonasList] = None,
    ) -> t.List[OverlapBasedScenario]:
        
        cluster_dict = knowledge_graph.find_direct_clusters(
            relationship_condition=lambda rel: (
                True if rel.type == "entities_overlap" else False
            ))

        valid_relationships = [rel for rel in knowledge_graph.relationships if rel.type == "entities_overlap"]

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
                    persona_concepts = await self.theme_persona_matching_prompt.generate(
                        data=prompt_input, llm=self.llm, callbacks=callbacks
                    )
                    overlapped_items = [list(item) for item in overlapped_items]
                    base_scenarios = self._prepare_seed_node_persona_mapping([key_node, node], overlapped_items, persona_concepts)
                    base_scenarios = sample_diverse_combinations(base_scenarios, num_sample_per_cluster)
                    scenarios.extend(base_scenarios)
                
        scenarios_to_return = []
        for scenario in scenarios:
            if len(scenarios_to_return) < n:
                scenarios_to_return.append(OverlapBasedScenario(
                    keyphrases=scenario["combination"],
                    nodes=scenario["nodes"],
                    style=scenario["style"],
                    length=scenario["length"],
                    persona=persona_list[scenario["persona"]],
                ))
                
        return scenarios_to_return
    
    async def _generate_sample(
        self, scenario: OverlapBasedScenario, callbacks: Callbacks
    ) -> SingleTurnSample:

        reference_context = self.make_contexts(scenario)
        prompt_input = QueryConditions(
            persona=scenario.persona,
            themes=scenario.keyphrases,
            context=reference_context,
            query_length=scenario.length.name,
            query_style=scenario.style.name,
        )
        response = await self.generate_query_reference_prompt.generate(
            data=prompt_input, llm=self.llm, callbacks=callbacks
        )
        return SingleTurnSample(
            user_input=response.query,
            reference=response.answer,
            reference_contexts=reference_context,
        )

    def make_contexts(self, scenario: OverlapBasedScenario) -> t.List[str]:

        contexts = []
        for node in scenario.nodes:
            context = f"{node.id}" + "\n\n" + node.properties.get("page_content", "")
            contexts.append(context)
            
        return contexts

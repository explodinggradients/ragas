from __future__ import annotations

import copy
import logging
import random
from turtle import pos
import typing as t
from dataclasses import dataclass
from itertools import product

import numpy as np

from ragas.dataset_schema import SingleTurnSample
from ragas.prompt import PydanticPrompt
from ragas.testset.graph import KnowledgeGraph, Node
from ragas.testset.synthesizers.base import BaseScenario, QueryLength, QueryStyle
from ragas.testset.synthesizers.base_query import QuerySynthesizer
from ragas.testset.persona import PersonasList
from ragas.testset.synthesizers.multihop.sampling import sample_diverse_combinations
from ragas.testset.synthesizers.prompts import (
    ConceptCombinationPrompt,
    ConceptsList,
    QueryAnswerGenerationPrompt,
    QueryConditions,
    ThemesList,
    ThemesPersonasInput,
    ThemesPersonasMatchingPrompt,
)

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks

logger = logging.getLogger(__name__)


class MultiHopAbstractQueryScenario(BaseScenario):
    """
    Scenario for multi-hop abstract queries.

    Attributes
    ----------
    theme: str
        The theme of the query.
    style: QueryStyle
        The style of the query.
    length: QueryLength
        The length of the query.
    """

    themes: t.List[str]


def get_child_nodes(node: Node, graph: KnowledgeGraph, level: int = 1) -> t.List[Node]:
    """
    Get the child nodes of a given node up to a specified level.

    Parameters
    ----------
    node : Node
        The node to get the children of.
    graph : KnowledgeGraph
        The knowledge graph containing the node.
    level : int
        The maximum level to which child nodes are searched.

    Returns
    -------
    List[Node]
        The list of child nodes up to the specified level.
    """
    children = []

    # Helper function to perform depth-limited search for child nodes
    def dfs(current_node: Node, current_level: int):
        if current_level > level:
            return
        for rel in graph.relationships:
            if rel.source == current_node and rel.type == "child":
                children.append(rel.target)
                dfs(rel.target, current_level + 1)

    # Start DFS from the initial node at level 0
    dfs(node, 1)

    return children


@dataclass
class MultiHopAbstractQuery(QuerySynthesizer):
    """
    Synthesizes abstract multi-hop queries from given knowledge graph.

    Attributes
    ----------
    """

    concept_combination_prompt: PydanticPrompt = ConceptCombinationPrompt()
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
                node_themes = [theme.lower() for theme in node.get_property("themes")]
                if node.get_property("themes") and any(
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
    ) -> t.List[MultiHopAbstractQueryScenario]:

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
            
            base_scenarios = self._prepare_seed_node_persona_mapping(nodes, concept_combination.combinations, persona_concepts)
            base_scenarios = sample_diverse_combinations(base_scenarios, num_sample_per_cluster)
            scenarios.extend(base_scenarios)
            
        random.shuffle(scenarios)

        return scenarios

    async def _generate_sample(
        self, scenario: MultiHopAbstractQueryScenario, callbacks: Callbacks
    ) -> SingleTurnSample:

        reference_context = self.make_contexts(scenario)
        prompt_input = QueryConditions(
            persona=scenario.persona,
            themes=scenario.themes,
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

    def make_contexts(self, scenario: MultiHopAbstractQueryScenario) -> t.List[str]:

        contexts = []
        for node in scenario.nodes:
            context = f"{node.id}" + "\n\n" + node.properties.get("page_content", "")
            contexts.append(context)

        return contexts

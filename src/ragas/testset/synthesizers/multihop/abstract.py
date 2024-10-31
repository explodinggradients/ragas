from __future__ import annotations

import logging
import random
import typing as t
from dataclasses import dataclass, field

from ragas.dataset_schema import SingleTurnSample
from ragas.executor import run_async_batch
from ragas.prompt import PydanticPrompt
from ragas.testset.graph import KnowledgeGraph, Node

from ragas.testset.synthesizers.base import BaseScenario, QueryLength, QueryStyle
from ragas.testset.synthesizers.base_query import QuerySynthesizer
from ragas.testset.synthesizers.prompts import Persona, ConceptCombinationPrompt, PersonaThemesMapping, PersonasList, ThemesPersonasMatchingPrompt, ConceptsList, ThemesList


if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks

logger = logging.getLogger(__name__)



@dataclass
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
    nodes: t.List[Node]
    themes: t.List[str]
    persona: Persona
    style: QueryStyle
    length: QueryLength


@dataclass
class MultiHopAbstractQuery(QuerySynthesizer):
    """
    Synthesizes abstract multi-hop queries from given knowledge graph.

    Attributes
    ----------
    """
    concept_combination_prompt: PydanticPrompt = ConceptCombinationPrompt()
    theme_persona_matching_prompt: PydanticPrompt = ThemesPersonasMatchingPrompt()
    
    
    async def _generate_scenarios(
        self, n: int, knowledge_graph: KnowledgeGraph, persona_list: PersonasList, callbacks: Callbacks, 
    ) -> t.List[MultiHopAbstractQueryScenario]:
        
        
        node_clusters = knowledge_graph.find_indirect_clusters(
            relationship_condition=lambda rel: (
                True if rel.get_property("summary_similarity") else False
            )
        )
        logger.info("found %d clusters", len(node_clusters))
        scenarios = []

        for cluster in node_clusters:
            
            cluster = list(cluster)[:3]
            themes = [node.properties.get("themes") for node in cluster]
            prompt_input = ConceptsList(lists_of_concepts=themes)
            concept_combination = await self.concept_combination_prompt.generate(data=prompt_input, llm=self.llm, callbacks=callbacks)
            flattened_themes = [theme for sublist in themes for theme in sublist]
            prompt_input = ThemesList(themes=flattened_themes)
            persona_concepts = await self.theme_persona_matching_prompt.generate(data=prompt_input, llm=self.llm, callbacks=callbacks)
            themes = [node.properties.get("themes", "").lower() for node in cluster]
            for combination in concept_combination:
                for persona, concept_list in persona_concepts.items():
                    concept_list = [c.lower() for c in concept_list]
                    indices = [idx for idx,theme in enumerate(themes) if any(concept.lower() in concept_list for concept in theme)]
                    nodes = [cluster[idx] for idx in indices]
                    if any(concept.lower() in concept_list for concept in combination):
                        persona = persona_list[persona]
                        for style, length in product(list(QueryStyle), list(QueryLength)):
                            scenarios.append(MultiHopAbstractQueryScenario(nodes=nodes, themes=themes, persona=persona, style=style, length=length))
               
        return scenarios         
        


    
    

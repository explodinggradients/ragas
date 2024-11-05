from __future__ import annotations

import logging
import random
import typing as t
from dataclasses import dataclass

import numpy as np

from ragas.dataset_schema import SingleTurnSample
from ragas.prompt import PydanticPrompt
from ragas.testset.graph import KnowledgeGraph
from ragas.testset.persona import Persona, PersonaList
from ragas.testset.synthesizers.base import (
    BaseScenario,
    BaseSynthesizer,
    QueryLength,
    QueryStyle,
    Scenario,
)
from ragas.testset.synthesizers.prompts import (
    ThemesPersonasInput,
    ThemesPersonasMatchingPrompt,
)
from ragas.testset.synthesizers.single_hop.prompts import (
    QueryAnswerGenerationPrompt,
    QueryCondition,
)

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
class SingleHopQuerySynthesizer(BaseSynthesizer[Scenario]):

    theme_persona_matching_prompt: PydanticPrompt = ThemesPersonasMatchingPrompt()
    generate_query_reference_prompt: PydanticPrompt = QueryAnswerGenerationPrompt()

    def prepare_combinations(
        self,
        node,
        terms: t.List[str],
        persona_list: PersonaList,
        persona_concepts,
    ) -> t.List[t.Dict[str, t.Any]]:

        sample = {"terms": terms, "node": node}
        valid_personas = []
        for persona, concepts in persona_concepts.mapping.items():
            concepts = [concept.lower() for concept in concepts]
            if any(term.lower() in concepts for term in terms):
                if persona_list[persona]:
                    valid_personas.append(persona_list[persona])
            sample["personas"] = valid_personas
            sample["styles"] = list(QueryStyle)
            sample["lengths"] = list(QueryLength)

        return [sample]

    def sample_combinations(self, data: t.List[t.Dict[str, t.Any]], num_samples):

        selected_samples = []
        node_term_set = set()

        all_combinations = []
        for entry in data:
            node = entry["node"]
            for term in entry["terms"]:
                for persona in entry["personas"]:
                    for style in entry["styles"]:
                        for length in entry["lengths"]:
                            all_combinations.append(
                                {
                                    "term": term,
                                    "node": node,
                                    "persona": persona,
                                    "style": style,
                                    "length": length,
                                }
                            )

        random.shuffle(all_combinations)
        for sample in all_combinations:
            if len(selected_samples) >= num_samples:
                break

            term = sample["term"]
            node = sample["node"]

            if (node, term) not in node_term_set:
                selected_samples.append(sample)
                node_term_set.add((node, term))
            elif len(selected_samples) < num_samples:
                selected_samples.append(sample)

        return [self.convert_to_scenario(sample) for sample in selected_samples]

    def convert_to_scenario(self, data: t.Dict[str, t.Any]) -> SingleHopScenario:

        return SingleHopScenario(
            term=data["term"],
            nodes=[data["node"]],
            persona=data["persona"],
            style=data["style"],
            length=data["length"],
        )

    async def _generate_scenarios(
        self,
        n: int,
        knowledge_graph: KnowledgeGraph,
        persona_list: t.List[Persona],
        callbacks: Callbacks,
    ) -> t.List[SingleHopScenario]:

        assert persona_list is not None, "Persona list is required for this synthesizer"
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

    async def _generate_sample(
        self, scenario: SingleHopScenario, callbacks: Callbacks
    ) -> SingleTurnSample:

        reference_context = self.make_contexts(scenario)
        prompt_input = QueryCondition(
            persona=scenario.persona,
            term=scenario.term,
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

    def make_contexts(self, scenario: SingleHopScenario) -> t.List[str]:

        contexts = []
        for node in scenario.nodes:
            context = f"{node.id}" + "\n\n" + node.properties.get("page_content", "")
            contexts.append(context)

        return contexts

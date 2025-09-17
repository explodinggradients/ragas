import typing as t

import pytest

from ragas.prompt import PydanticPrompt
from ragas.testset.persona import Persona
from ragas.testset.synthesizers.base import QueryLength, QueryStyle
from ragas.testset.synthesizers.multi_hop.abstract import (
    MultiHopAbstractQuerySynthesizer,
)
from ragas.testset.synthesizers.multi_hop.prompts import (
    ConceptCombinations,
    ConceptsList,
)
from ragas.testset.synthesizers.prompts import PersonaThemesMapping, ThemesPersonasInput
from tests.unit.test_knowledge_graph_clusters import (
    build_knowledge_graph,
    create_chain_of_similarities,
    create_document_and_child_nodes,
)


class MockConceptCombinationPrompt(PydanticPrompt):
    async def generate(self, data: ConceptsList, llm, callbacks=None):
        concepts: t.List[t.List[str]] = data.lists_of_concepts
        max_combinations: int = data.max_combinations
        return ConceptCombinations(combinations=concepts[:max_combinations])


class MockThemePersonaMatchingPrompt(PydanticPrompt):
    async def generate(self, data: ThemesPersonasInput, llm, callbacks=None):
        themes: t.List[str] = data.themes
        personas: t.List[Persona] = data.personas
        return PersonaThemesMapping(
            mapping={persona.name: themes for persona in personas}
        )


def _assert_scenario_properties(
    scenarios: list[t.Any], personas: list[Persona]
) -> None:
    """Validate scenario has the expected properties."""
    for scenario in scenarios:
        assert hasattr(scenario, "nodes")
        assert hasattr(scenario, "persona")
        assert hasattr(scenario, "style")
        assert hasattr(scenario, "length")
        assert hasattr(scenario, "combinations")

        # Check that the persona is from our list
        assert scenario.persona in personas
        assert scenario.style in QueryStyle
        assert scenario.length in QueryLength
        # Check that the document node was eliminated and replaced with its children
        for node in scenario.nodes:
            assert str(node.id) in [
                "2",
                "3",
                "4",
                "5",
                "1_1",
                "1_2",
                "1_1_1",
                "1_1_2",
                "1_1_3",
            ]
        # Check that the combinations are from the themes we defined
        for item in scenario.combinations:
            assert item in [
                "T_2",
                "T_3",
                "T_4",
                "T_5",
                "T_1_1",
                "T_1_2",
                "T_1_1_1",
                "T_1_1_2",
                "T_1_1_3",
            ]


@pytest.mark.asyncio
async def test_generate_scenarios(fake_llm):
    """Test the _generate_scenarios method of MultiHopAbstractQuerySynthesizer."""
    nodes, relationships = create_document_and_child_nodes()
    sim_nodes, sim_relationships = create_chain_of_similarities(nodes[0], node_count=3)
    branch_nodes, branch_relationships = create_chain_of_similarities(
        sim_nodes[1], node_count=4
    )
    nodes.extend(sim_nodes[1:])
    nodes.extend(branch_nodes[1:])
    relationships.extend(sim_relationships)
    relationships.extend(branch_relationships)
    kg = build_knowledge_graph(nodes, relationships)

    personas = [
        Persona(
            name="Researcher",
            role_description="Researcher interested in the latest advancements in AI.",
        ),
        Persona(
            name="Engineer",
            role_description="Engineer interested in the latest advancements in AI.",
        ),
    ]

    synthesizer = MultiHopAbstractQuerySynthesizer(llm=fake_llm)

    # Replace the prompts with mock versions
    synthesizer.concept_combination_prompt = MockConceptCombinationPrompt()
    synthesizer.theme_persona_matching_prompt = MockThemePersonaMatchingPrompt()

    num_nodes = len(kg.nodes)
    for n in range(1, num_nodes + 3):
        scenarios = await synthesizer._generate_scenarios(
            n=n,
            knowledge_graph=kg,
            persona_list=personas,
            callbacks=None,
        )

        # Assert we got the expected number of scenarios
        # Must be a range to compensate for num_sample_per_cluster rounding
        assert n <= len(scenarios) <= n + 2, (
            f"Expected {n} or {n + 1} scenarios, got {len(scenarios)}"
        )
        _assert_scenario_properties(scenarios, personas)

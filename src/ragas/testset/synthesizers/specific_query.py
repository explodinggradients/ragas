from __future__ import annotations

import random
import typing as t
from dataclasses import dataclass, field

from ragas.dataset_schema import SingleTurnSample
from ragas.prompt import PydanticPrompt
from ragas.testset.graph import KnowledgeGraph, NodeType

from .base import BaseScenario, QueryLength, QueryStyle
from .base_query import QuerySynthesizer
from .prompts import SpecificQuery, SpecificQuestionInput

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks


class SpecificQueryScenario(BaseScenario):
    """
    Represents a scenario for generating specific queries.
    Also inherits attributes from [BaseScenario][ragas.testset.synthesizers.base.BaseScenario].

    Attributes
    ----------
    keyphrase : str
        The keyphrase of the specific query scenario.
    """

    keyphrase: str


@dataclass
class SpecificQuerySynthesizer(QuerySynthesizer):
    """
    Synthesizes specific queries by choosing specific chunks and generating a
    keyphrase from them and then generating queries based on that.

    Attributes
    ----------
    generate_query_prompt : PydanticPrompt
        The prompt used for generating the query.
    """

    generate_query_prompt: PydanticPrompt = field(default_factory=SpecificQuery)

    async def _generate_scenarios(
        self, n: int, knowledge_graph: KnowledgeGraph, callbacks: Callbacks
    ) -> t.List[SpecificQueryScenario]:
        # filter out nodes that have keyphrases
        nodes = []
        for node in knowledge_graph.nodes:
            if (
                node.type == NodeType.CHUNK
                and node.get_property("keyphrases") is not None
                and node.get_property("keyphrases") != []
            ):
                nodes.append(node)

        # sample nodes and keyphrases
        sampled_nodes = random.choices(nodes, k=n)
        sampled_keyphrases = []
        for node in sampled_nodes:
            sampled_keyphrases_per_node = set()
            keyphrases = node.get_property("keyphrases")
            unused_keyphrases = list(set(keyphrases) - sampled_keyphrases_per_node)
            if unused_keyphrases:
                sampled_keyphrases.append(random.choice(unused_keyphrases))
            else:
                sampled_keyphrases.append(random.choice(keyphrases))

        # sample query styles and lengths
        query_styles = random.choices(list(QueryStyle), k=n)
        query_lengths = random.choices(list(QueryLength), k=n)

        scenarios = []
        for node, keyphrase, style, length in zip(
            sampled_nodes, sampled_keyphrases, query_styles, query_lengths
        ):
            scenarios.append(
                SpecificQueryScenario(
                    nodes=[node], keyphrase=keyphrase, style=style, length=length
                )
            )
        return scenarios

    async def _generate_sample(
        self, scenario: SpecificQueryScenario, callbacks: t.Optional[Callbacks] = None
    ) -> SingleTurnSample:
        query = await self.generate_query_prompt.generate(
            data=SpecificQuestionInput(
                title=scenario.nodes[0].get_property("title") or "",
                keyphrase=scenario.keyphrase,
                text=scenario.nodes[0].get_property("page_content") or "",
            ),
            llm=self.llm,
            callbacks=callbacks,
        )

        query_text = query.text
        if not await self.critic_query(query_text, callbacks):
            query_text = await self.modify_query(query_text, scenario, callbacks)

        reference = await self.generate_reference(query_text, scenario, callbacks)

        reference_contexts = []
        for node in scenario.nodes:
            if node.get_property("page_content") is not None:
                reference_contexts.append(node.get_property("page_content"))

        return SingleTurnSample(
            user_input=query_text,
            reference=reference,
            reference_contexts=reference_contexts,
        )

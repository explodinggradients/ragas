from __future__ import annotations

import random
import typing as t
from dataclasses import dataclass, field

from ragas.dataset_schema import SingleTurnSample
from ragas.experimental.testset.graph import KnowledgeGraph, NodeType
from ragas.prompt import PydanticPrompt

from .base import BaseScenario, UserInputLength, UserInputStyle
from .base_qa import QASimulator
from .prompts import SpecificQuestion, SpecificQuestionInput

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks


class SpecificQuestionScenario(BaseScenario):
    keyphrase: str


@dataclass
class SpecificQASimulator(QASimulator):
    generate_question_prompt: PydanticPrompt = field(default_factory=SpecificQuestion)

    async def _generate_scenarios(
        self, n: int, knowledge_graph: KnowledgeGraph, callbacks: Callbacks
    ) -> t.List[SpecificQuestionScenario]:
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

        # sample question styles and lengths
        question_styles = random.choices(list(UserInputStyle), k=n)
        question_lengths = random.choices(list(UserInputLength), k=n)

        scenarios = []
        for node, keyphrase, style, length in zip(
            sampled_nodes, sampled_keyphrases, question_styles, question_lengths
        ):
            scenarios.append(
                SpecificQuestionScenario(
                    nodes=[node], keyphrase=keyphrase, style=style, length=length
                )
            )
        return scenarios

    async def _generate_sample(
        self, scenario: SpecificQuestionScenario, callbacks: Callbacks
    ) -> SingleTurnSample:
        question = await self.generate_question_prompt.generate(
            data=SpecificQuestionInput(
                title=scenario.nodes[0].get_property("title") or "",
                keyphrase=scenario.keyphrase,
                text=scenario.nodes[0].get_property("page_content") or "",
            ),
            llm=self.llm,
            callbacks=callbacks,
        )

        question_text = question.text
        if not await self.critic_question(question_text, callbacks):
            question_text = await self.modify_question(
                question_text, scenario, callbacks
            )

        reference = await self.generate_answer(question_text, scenario, callbacks)

        reference_contexts = []
        for node in scenario.nodes:
            if node.get_property("page_content") is not None:
                reference_contexts.append(node.get_property("page_content"))

        return SingleTurnSample(
            user_input=question_text,
            reference=reference,
            reference_contexts=reference_contexts,
        )

import json
import logging
import typing as t
from dataclasses import dataclass, field

import numpy as np
from langchain_core.documents import Document as LCDocument
from ragas_experimental.testset.graph import Node
from ragas_experimental.testset.questions.base import (
    DEFAULT_DISTRIBUTION,
    QAC,
    QAGenerator,
    QuestionLength,
    QuestionStyle,
)
from ragas_experimental.testset.questions.prompts import (
    critic_question,
    order_sections_by_relevance,
    question_answering,
    specific_question_from_keyphrase,
)
from ragas_experimental.testset.questions.queries import CHILD_NODES_QUERY
from ragas_experimental.testset.utils import rng

from ragas.executor import Executor
from ragas.llms.prompt import Prompt

logger = logging.getLogger(__name__)


@dataclass
class SpecificQA(QAGenerator):
    name: str = "SpecificQuestion"
    generate_question_prompt: Prompt = field(
        default_factory=lambda: specific_question_from_keyphrase
    )
    generate_answer_prompt: Prompt = field(default_factory=lambda: question_answering)
    critic_question_prompt: Prompt = field(default_factory=lambda: critic_question)
    order_sections_prompt: Prompt = field(
        default_factory=lambda: order_sections_by_relevance
    )

    """
    specic questions from particular sections of particular document
    for example, what are model architecture details used in OpenMoE paper?
    """

    async def generate_questions(
        self, query, kwargs, distribution=DEFAULT_DISTRIBUTION, num_samples=5
    ):
        assert self.llm is not None, "LLM is not initialized"
        query = query or CHILD_NODES_QUERY
        kwargs = kwargs or {}
        nodes = self.query_nodes(query, kwargs)
        num_nodes = min(num_samples, len(nodes))
        nodes = rng.choice(nodes, size=num_nodes, replace=False)
        seed_per_results = num_samples // len(nodes)
        reminder = num_samples - seed_per_results * num_nodes
        seeds = [seed_per_results] * num_nodes
        seeds[-1] += reminder

        nodes_and_keyphraes = []
        for node, seed in zip(nodes, seeds):
            seperators = [
                rel.properties["seperator"]
                for rel in node.relationships
                if rel.label == "child"
            ]
            if len(seperators) > 1:
                p_vlaue = self.order_sections_prompt.format(sections=seperators)
                output = await self.llm.generate(p_vlaue)
                output = json.loads(output.generations[0][0].text)
                # TODO: make sure that prompt does not modify the seperator. Ideally ouput ordering by index
                headings_array = (
                    np.array(output.get("high"))
                    if output.get("high")
                    else np.array(list(output.values()))
                )
                selected_heading = rng.choice(headings_array, size=seed)
            else:
                # TODO: inspect and handle better
                selected_heading = seperators[0:1]

            target_nodes = [
                relation.target
                for relation in node.relationships
                if relation.label == "child"
                and relation.source.id == node.id
                and any(
                    text in relation.properties["seperator"]
                    for text in selected_heading
                )
            ]

            keyphrases = [
                node.properties["metadata"]["keyphrases"] for node in target_nodes
            ]
            keyphrases = list(
                set([phrase for phrases in keyphrases for phrase in phrases])
            )
            keyphrases = rng.choice(np.array(keyphrases), size=seed)
            nodes_and_keyphraes.extend(
                [(target_nodes, keyphrase) for keyphrase in keyphrases]
            )

        exec = Executor(
            desc="Generating",
            keep_progress_bar=True,
            raise_exceptions=True,
            run_config=None,
        )

        index = 0
        for dist, prob in distribution.items():
            style, length = dist
            for i in range(int(prob * num_samples)):
                exec.submit(
                    self.generate_question,
                    nodes_and_keyphraes[index][0],
                    style,
                    length,
                    {"keyphrase": nodes_and_keyphraes[index][1]},
                )
                index += 1

        remaining_size = num_samples - index
        if remaining_size != 0:
            choices = np.array(distribution.keys())
            prob = np.array(distribution.values())
            random_distribution = rng.choice(choices, p=prob, size=remaining_size)
            for dist in random_distribution:
                style, length = dist
                exec.submit(
                    self.generate_question,
                    nodes_and_keyphraes[index][0],
                    style,
                    length,
                    {"keyphrase": nodes_and_keyphraes[index][1]},
                )
                index += 1

        return exec.results()

    async def generate_question(
        self,
        nodes: t.List[Node],
        style: QuestionStyle,
        length: QuestionLength,
        kwargs: t.Optional[dict] = None,
    ) -> QAC:
        assert self.llm is not None, "LLM is not initialized"
        assert self.embedding is not None, "Embedding is not initialized"
        kwargs = kwargs or {}

        keyphrase = kwargs["keyphrase"]
        title = nodes[0].properties["metadata"]["title"]
        text = "\n\n".join([node.properties["page_content"] for node in nodes])
        try:
            source = self.retrieve_chunks(nodes)
            if source:
                p_value = self.generate_question_prompt.format(
                    title=title, keyphrase=keyphrase, text=text
                )
                question = await self.llm.generate(p_value)
                question = question.generations[0][0].text

                critic_verdict = await self.critic_question(question)
                if critic_verdict:
                    question = await self.modify_question(question, style, length)
                    answer = await self.generate_answer(question, source)
                    return QAC(
                        question=question,
                        answer=answer,
                        source=source,
                        name=self.name,
                        style=style,
                        length=length,
                    )
                else:
                    logger.warning("Critic rejected the question: %s", question)
                    return QAC()
            else:
                logger.warning("Failed to retrieve chunks for nodes: %s", nodes)
                return QAC()

        except Exception as e:
            logging.warning("Failed to generate question: %s", e)
            raise e

    async def critic_question(self, question: str) -> bool:
        assert self.llm is not None, "LLM is not initialized"
        output = await self.llm.generate(critic_question.format(question=question))
        output = json.loads(output.generations[0][0].text)
        return all(score >= 2 for score in output.values())

    def retrieve_chunks(self, nodes: t.List[Node], kwargs: dict | None = None) -> t.Any:
        assert self.llm is not None, "LLM is not initialized"
        documents = [
            LCDocument(
                page_content=node.properties["page_content"],
                metadata=node.properties["metadata"],
            )
            for node in nodes
        ]
        return documents

    async def generate_answer(self, question: str, chunks: t.List[LCDocument]) -> t.Any:
        assert self.llm is not None, "LLM is not initialized"
        text = "\n\n".join([chunk.page_content for chunk in chunks])
        output = await self.llm.generate(
            self.generate_answer_prompt.format(question=question, text=text)
        )
        return output.generations[0][0].text

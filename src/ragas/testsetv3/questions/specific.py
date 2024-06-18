import json
import logging
import typing as t
from dataclasses import dataclass, field

import numpy as np
from langchain_core.documents import Document as LCDocument

from ragas.llms.prompt import Prompt
from ragas.testsetv3.graph import Node
from ragas.testsetv3.questions.base import QAC, QAGenerator
from ragas.testsetv3.questions.prompts import (
    critic_question,
    order_sections_by_relevance,
    question_answering,
    specific_question_from_keyphrase,
)
from ragas.testsetv3.utils import rng

logger = logging.getLogger(__name__)


@dataclass
class SpecificQuestion(QAGenerator):
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

    async def generate_question(
        self, query: str | None = None, kwargs: dict | None = None
    ) -> QAC:
        assert self.llm is not None, "LLM is not initialized"
        assert self.embedding is not None, "Embedding is not initialized"

        query = """
        {{
        filterNodes(label: DOC, level : LEVEL_0) {{
            id
            label
            properties
            relationships(label: "{label}") {{
            label
            properties
            target {{
                id
                label
                properties
            }}
            source {{
                id
            }}
            }}
        }}
        }}
        """
        kwargs = {
            "label": "contains",
        }
        result_nodes = self.query_nodes(query, kwargs)
        current_node = self.get_random_node(result_nodes)

        seperators = [
            rel.properties["seperator"]
            for rel in current_node[0].relationships
            if rel.label == "contains"
        ]
        if len(seperators) > 1:
            p_vlaue = self.order_sections_prompt.format(sections=seperators)
            output = await self.llm.generate(p_vlaue)
            output = json.loads(output.generations[0][0].text)
            # TODO: make sure that prompt does not modify the seperator. Ideally ouput ordering by index
            headings_array = np.array(output.get("high"))
            selected_heading = rng.choice(headings_array, size=1)[0:1]
        else:
            # TODO: inspect and handle better
            selected_heading = seperators[0:1]

        nodes = [
            relation.target
            for relation in current_node[0].relationships
            if relation.label == "contains"
            and relation.source.id == current_node[0].id
            and any(
                text in relation.properties["seperator"] for text in selected_heading
            )
        ]

        if not nodes:
            return QAC()

        keyphrases = [node.properties["metadata"]["keyphrases"] for node in nodes]
        keyphrases = list(set([phrase for phrases in keyphrases for phrase in phrases]))
        keyphrase = rng.choice(np.array(keyphrases), size=1)[0]
        title = current_node[0].properties["metadata"]["title"]
        text = nodes[0].properties["page_content"]
        p_value = self.generate_question_prompt.format(
            title=title, keyphrase=keyphrase, text=text
        )
        question = await self.llm.generate(p_value)
        question = question.generations[0][0].text

        critic_verdict = await self.critic_question(question)
        if critic_verdict:
            source = self.retrieve_chunks(question, nodes)
            question = await self.modify_question(question)
            answer = await self.generate_answer(question, source)
            return QAC(
                question=question,
                answer=answer,
                source=source,
                name=self.name,
                style=self.style,
                length=self.length,
            )
        else:
            logger.warning("Critic rejected the question: %s", question)
            return QAC()

    async def critic_question(self, question: str) -> bool:
        assert self.llm is not None, "LLM is not initialized"
        output = await self.llm.generate(critic_question.format(question=question))
        output = json.loads(output.generations[0][0].text)
        return all(score >= 2 for score in output.values())

    def retrieve_chunks(
        self, question: str, nodes: t.List[Node], kwargs: dict | None = None
    ) -> t.Any:
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

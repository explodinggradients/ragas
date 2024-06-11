import json
import logging
import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import tiktoken
from graphene.types.schema import Schema
from langchain.utils.math import cosine_similarity
from langchain_core.documents import Document as LCDocument

from ragas.embeddings import BaseRagasEmbeddings
from ragas.llms.base import BaseRagasLLM
from ragas.llms.prompt import Prompt
from ragas.testsetv3.graph import Node, Relationship
from ragas.testsetv3.query_prompts import (
    abstract_question_from_theme,
    common_theme_from_summaries,
    common_topic_from_keyphrases,
    comparative_question,
    critic_question,
    question_answering,
)
from ragas.testsetv3.utils import rng

logger = logging.getLogger(__name__)


@dataclass
class QAC:
    question: str
    answer: str
    source: t.List[str]


@dataclass
class QueryGenerator(ABC):
    llm: BaseRagasLLM
    embedding: BaseRagasEmbeddings
    schema: Schema
    nodes: t.List[Node]
    relationships: t.List[Relationship]

    @abstractmethod
    def generate_question(
        self, query: t.Optional[str] = None, kwargs: t.Optional[dict] = None
    ) -> t.Any:
        pass

    @abstractmethod
    def critic_question(self, query: str) -> bool:
        pass

    @abstractmethod
    def generate_answer(self, query: str) -> t.Any:
        pass

    def query_nodes(self, query: str, kwargs) -> t.Any:
        query = query.format(**kwargs)
        results = self.schema.execute(
            query, context={"nodes": self.nodes, "relationships": self.relationships}
        )
        if results.errors:
            raise Exception(results.errors)
        if results.data is None:
            logger.warning("result for %s is None", query)
            return None
        return results.data


@dataclass
class AbstractQueries(QueryGenerator):
    generate_question_prompt: Prompt = field(
        default_factory=lambda: abstract_question_from_theme
    )
    critic_question_prompt: Prompt = field(default_factory=lambda: critic_question)
    generate_answer_prompt: Prompt = field(default_factory=lambda: question_answering)
    generate_common_theme_prompt: Prompt = field(
        default_factory=lambda: common_theme_from_summaries
    )

    def get_random_node(self, nodes) -> t.List[Node]:
        nodes = [node for node in nodes if node.relationships]
        nodes_weights = np.array(
            [json.loads(node.properties).get("chances", 0) for node in nodes]
        )
        if all(nodes_weights == 0):
            nodes_weights = np.ones(len(nodes_weights))
        nodes_weights = nodes_weights / sum(nodes_weights)
        return rng.choice(np.array(nodes), p=nodes_weights, size=1).tolist()

    async def generate_question(
        self, query: t.Optional[str] = None, kwargs: t.Optional[dict] = None
    ) -> t.Any:
        if query is None:
            query = """
            {{
            filterNodes(label: DOC) {{
                id
                label
                properties
                relationships(label: "{label}", propertyKey: "{property}", propertyValue: "{value}", comparison: "{comparison}") {{
                label
                properties
                target {{
                    id
                    label
                    properties
                }}
                }}
            }}
            }}
            """
        if kwargs is None:
            kwargs = {
                "label": "summary_similarity",
                "property": "score",
                "value": 0.5,
                "comparison": "gt",
            }

        results = self.query_nodes(query, kwargs)
        if results is None:
            return None

        result_nodes = [Node(**item) for item in results["filterNodes"]]
        current_nodes = self.get_random_node(result_nodes)

        related_nodes = [
            Node(**rel["target"])
            for rel in current_nodes[0].relationships
            if rel["target"]["label"] == "DOC"
        ]
        if not related_nodes:
            return None
        current_nodes.extend(related_nodes)
        summaries = [
            json.loads(item.properties)["metadata"]["summary"] for item in current_nodes
        ]
        summaries = "\n".join(
            [f"{i+1}. {summary}" for i, summary in enumerate(summaries)]
        )
        common_theme = await self.llm.generate(
            self.generate_common_theme_prompt.format(summaries=summaries)
        )
        common_theme = common_theme.generations[0][0].text
        abstract_question = await self.llm.generate(
            self.generate_question_prompt.format(
                theme=common_theme, summaries=summaries
            )
        )
        abstract_question = abstract_question.generations[0][0].text
        critic_verdict = await self.critic_question(abstract_question)
        if critic_verdict:
            answer, source = await self.generate_answer(
                abstract_question, current_nodes
            )
            return QAC(question=abstract_question, answer=answer, source=source)
        else:
            logger.warning("Critic rejected the question: %s", abstract_question)

    async def critic_question(self, query: str) -> bool:
        output = await self.llm.generate(critic_question.format(question=query))
        output = json.loads(output.generations[0][0].text)
        return all(score >= 2 for score in output.values())

    async def generate_answer(
        self, question: str, nodes: t.List[Node], max_tokens=4000
    ) -> t.Any:
        node_ids = [node.id for node in nodes]
        node_ids = json.dumps(node_ids)
        query = """
        {{
        filterNodes(ids: {node_ids}) {{
            id
            label
            properties
            relationships(label: "contains") {{
            label
            properties
            target {{
                id
                label
                properties
            }}
            }}
        }}
        }}
        """
        results = self.query_nodes(query, {"node_ids": node_ids})
        if results is None:
            return None
        nodes = [Node(**node) for node in results["filterNodes"]]
        summary_chunks = [
            json.loads(node.properties)["metadata"]["summary"] for node in nodes
        ]

        # query to get all child nodes of nodes
        nodes = [
            Node(**relationship["target"])
            for node in nodes
            for relationship in node.relationships
        ]
        chunks = [
            LCDocument(
                page_content=json.loads(node.properties)["page_content"],
                metadata=json.loads(node.properties)["metadata"],
            )
            for node in nodes
        ]
        chunks_embeddings = [
            json.loads(node.properties)["metadata"]["page_content_embedding"]
            for node in nodes
        ]
        question_embedding = await self.embedding.aembed_query(question)
        similarity_matrix = cosine_similarity([question_embedding], chunks_embeddings)
        most_similar = np.flip(np.argsort(similarity_matrix[0]))
        ranked_chunks = [chunks[i] for i in most_similar]
        model_name = self.llm.langchain_llm.model_name or "gpt-2"
        enc = tiktoken.encoding_for_model(model_name)
        ranked_chunks_length = [
            len(enc.encode(chunk.page_content)) for chunk in ranked_chunks
        ]
        ranked_chunks_length = np.cumsum(ranked_chunks_length)
        index = np.argmax(np.argwhere(np.cumsum(ranked_chunks_length) < max_tokens)) + 1
        ranked_chunks = ranked_chunks[:index]
        # TODO : add title+summary of each node + title + content from most relevant chunk
        text = (
            "\n\n".join(summary_chunks)
            + "\n\n"
            + "\n\n".join([chunk.page_content for chunk in ranked_chunks])
        )
        output = await self.llm.generate(
            self.generate_answer_prompt.format(question=question, text=text)
        )
        answer = output.generations[0][0].text

        return (answer, ranked_chunks + summary_chunks)


@dataclass
class ComparitiveAbstractQueries(AbstractQueries):
    common_topic_prompt: Prompt = field(
        default_factory=lambda: common_topic_from_keyphrases
    )
    generate_question_prompt: Prompt = field(
        default_factory=lambda: comparative_question
    )

    async def generate_question(
        self, query: t.Optional[str] = None, kwargs: t.Optional[dict] = None
    ) -> t.Any:
        query = """
        {{
        filterNodes(label: DOC) {{
            id
            label
            properties
            relationships(label: "{label}", propertyKey: "{property}", propertyValue: "{value}", comparison: "{comparison}") {{
            label
            properties
            target {{
                id
                label
                properties
            }}
            }}
        }}
        }}
        """
        kwargs = {
            "label": "jaccard_over_keyphrases",
            "property": "score",
            "value": 0.2,
            "comparison": "gt",
        }
        result_nodes = self.query_nodes(query, kwargs)
        result_nodes = [Node(**node) for node in result_nodes["filterNodes"]]
        current_nodes = self.get_random_node(result_nodes)

        indices = np.flip(
            np.argsort(
                [
                    json.loads(rel["properties"])["score"]
                    for rel in current_nodes[0].relationships
                ]
            )
        )[:3]
        related_nodes = [
            Node(**current_nodes[0].relationships[i]["target"]) for i in indices
        ]
        current_nodes.extend(related_nodes)

        summaries = [
            json.loads(item.properties)["metadata"]["summary"] for item in current_nodes
        ]
        summaries = "\n".join(
            [f"{i+1}. {summary}" for i, summary in enumerate(summaries)]
        )
        common_theme = await self.llm.generate(
            self.generate_common_theme_prompt.format(summaries=summaries)
        )
        common_theme = common_theme.generations[0][0].text

        keyphrases = [
            json.loads(node.properties)["metadata"]["keyphrases"]
            for node in current_nodes
        ]
        keyphrases = [phrase for phrases in keyphrases for phrase in phrases]
        comparison_topic = await self.llm.generate(
            self.common_topic_prompt.format(theme=common_theme, keyphrases=keyphrases)
        )
        comparison_topic = json.loads(comparison_topic.generations[0][0].text)[0]

        question = await self.llm.generate(
            self.generate_question_prompt.format(
                theme=common_theme, topic=comparison_topic
            )
        )
        question = question.generations[0][0].text

        return question


class SpecificQuestion(QueryGenerator):

    """
    specic questions from particular sections of particular document
    for example, what are model architecture details used in OpenMoE paper?
    """

    async def generate_question(self, query: str) -> t.Any:
        pass

    async def critic_question(self, query: str) -> bool:
        pass

    async def generate_answer(self, query: str) -> t.Any:
        pass

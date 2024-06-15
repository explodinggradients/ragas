import json
import logging
import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import tiktoken
from langchain.utils.math import cosine_similarity
from langchain_core.documents import Document as LCDocument

from ragas.embeddings import BaseRagasEmbeddings, embedding_factory
from ragas.llms.base import BaseRagasLLM, llm_factory
from ragas.llms.prompt import Prompt
from ragas.testsetv3.graph import Node, Relationship
from ragas.testsetv3.graph import schema as myschema
from ragas.testsetv3.query_prompts import (
    abstract_question_from_theme,
    common_theme_from_summaries,
    common_topic_from_keyphrases,
    comparative_question,
    critic_question,
    order_sections_by_relevance,
    question_answering,
    question_modification,
)
from ragas.testsetv3.utils import rng

logger = logging.getLogger(__name__)


class QuestionLength(Enum):
    LONG = "long"
    MEDIUM = "medium"
    SHORT = "short"


class QuestionStyle(Enum):
    MISSPELLED = "Misspelled queries"
    PERFECT_GRAMMAR = "Perfect grammar"
    POOR_GRAMMAR = "Poor grammar"
    WEB_SEARCH_LIKE = "Web search like queries"


@dataclass
class QAC:
    question: t.Optional[str] = None
    answer: t.Optional[str] = None
    source: t.Optional[t.List[LCDocument]] = None
    name: t.Optional[str] = None
    style: t.Optional[QuestionStyle] = None
    length: t.Optional[QuestionLength] = None


@dataclass
class QAGenerator(ABC):
    nodes: t.List[Node]
    relationships: t.List[Relationship]

    llm: t.Optional[BaseRagasLLM] = None
    embedding: t.Optional[BaseRagasEmbeddings] = None
    name: t.Optional[str] = None
    style: QuestionStyle = QuestionStyle.PERFECT_GRAMMAR
    length: QuestionLength = QuestionLength.MEDIUM
    question_modification_prompt: Prompt = field(
        default_factory=lambda: question_modification
    )

    def __post_init__(self):
        self.llm = self.llm or llm_factory()
        self.embedding = self.embedding or embedding_factory()

    @abstractmethod
    def generate_question(
        self, query: t.Optional[str] = None, kwargs: t.Optional[dict] = None
    ) -> t.Any:
        pass

    @abstractmethod
    def critic_question(self, question: str) -> bool:
        pass

    @abstractmethod
    def generate_answer(self, question: str, chunks: t.List[LCDocument]) -> t.Any:
        pass

    @abstractmethod
    def retrieve_chunks(
        self, question: str, nodes: t.List[Node], kwargs: t.Optional[dict] = None
    ) -> t.Any:
        pass

    async def modify_question(self, question: str) -> str:
        question_modification_prompt_ = self.question_modification_prompt
        examples = [
            example
            for example in self.question_modification_prompt.examples
            if example["style"] == self.style.value
            and example["length"] == self.length.value
        ]
        question_modification_prompt_.examples = examples
        p_value = question_modification_prompt_.format(
            question=question, style=self.style.value, length=self.length.value
        )
        question = await self.llm.generate(p_value)
        return question.generations[0][0].text

    def query_nodes(self, query: str, kwargs) -> t.Any:
        query = query.format(**kwargs)
        results = myschema.execute(
            query, context={"nodes": self.nodes, "relationships": self.relationships}
        )
        if results.errors:
            raise Exception(results.errors)
        if results.data is None:
            logger.warning("result for %s is None", query)
            return None
        return results.data

    def get_random_node(self, nodes) -> t.List[Node]:
        nodes = [node for node in nodes if node.relationships]
        nodes_weights = np.array(
            [json.loads(node.properties).get("chances", 0) for node in nodes]
        )
        if all(nodes_weights == 0):
            nodes_weights = np.ones(len(nodes_weights))
        nodes_weights = nodes_weights / sum(nodes_weights)
        return rng.choice(np.array(nodes), p=nodes_weights, size=1).tolist()


@dataclass
class AbtractQA(QAGenerator):
    name: str = "AbstractQA"
    generate_question_prompt: Prompt = field(
        default_factory=lambda: abstract_question_from_theme
    )
    critic_question_prompt: Prompt = field(default_factory=lambda: critic_question)
    generate_answer_prompt: Prompt = field(default_factory=lambda: question_answering)
    generate_common_theme_prompt: Prompt = field(
        default_factory=lambda: common_theme_from_summaries
    )

    async def generate_question(
        self, query: t.Optional[str] = None, kwargs: t.Optional[dict] = None
    ) -> QAC:
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
            return QAC()
        else:
            if not results["filterNodes"]:
                return QAC()

        result_nodes = [Node(**item) for item in results["filterNodes"]]
        current_nodes = self.get_random_node(result_nodes)

        related_nodes = [
            Node(**rel["target"])
            for rel in current_nodes[0].relationships
            if rel["target"]["label"] == "DOC"
        ]
        if not related_nodes:
            return QAC()
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
            source = await self.retrieve_chunks(abstract_question, current_nodes)
            abstract_question = await self.modify_question(abstract_question)
            answer = await self.generate_answer(abstract_question, source)
            return QAC(
                question=abstract_question,
                answer=answer,
                source=source,
                name=self.name,
                style=self.style,
                length=self.length,
            )
        else:
            logger.warning("Critic rejected the question: %s", abstract_question)
            return QAC()

    async def critic_question(self, question: str) -> bool:
        output = await self.llm.generate(critic_question.format(question=question))
        output = json.loads(output.generations[0][0].text)
        return all(score >= 2 for score in output.values())

    async def retrieve_chunks(
        self, question: str, nodes: t.List[Node], kwargs: t.Optional[dict] = None
    ) -> t.List[LCDocument]:
        kwargs = kwargs or {}
        max_tokens = kwargs.get("max_tokens", 4024)
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
        output_documents = [
            LCDocument(
                page_content=json.loads(node.properties)["metadata"]["summary"],
                metadata={"source": json.loads(node.properties)["metadata"]["source"]},
            )
            for node in nodes
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
        output_documents.extend(ranked_chunks[:index])

        return output_documents

    async def generate_answer(
        self,
        question: str,
        chunks: t.List[LCDocument],
    ) -> str:
        # TODO : add title+summary of each node + title + content from most relevant chunk
        text = "\n\n".join([chunk.page_content for chunk in chunks])
        output = await self.llm.generate(
            self.generate_answer_prompt.format(question=question, text=text)
        )
        return output.generations[0][0].text


@dataclass
class ComparitiveAbtractQA(AbtractQA):
    name: str = "ComparitiveAbtractQA"
    common_topic_prompt: Prompt = field(
        default_factory=lambda: common_topic_from_keyphrases
    )
    generate_question_prompt: Prompt = field(
        default_factory=lambda: comparative_question
    )

    async def generate_question(
        self, query: t.Optional[str] = None, kwargs: t.Optional[dict] = None
    ) -> QAC:
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


@dataclass
class SpecificQuestion(QAGenerator):
    name: str = "SpecificQuestion"
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
        query = """
        {{
        filterNodes(label: DOC) {{
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
            }}
        }}
        }}
        """
        kwargs = {
            "label": "contains",
        }
        result_nodes = self.query_nodes(query, kwargs)
        result_nodes = [Node(**node) for node in result_nodes["filterNodes"]]
        current_node = self.get_random_node(result_nodes)

        headings = json.loads(current_node[0].properties)["metadata"]["headlines"]
        p_vlaue = self.order_sections_prompt.format(sections=list(headings.keys()))
        output = await self.llm.generate(p_vlaue)
        output = json.loads(output.generations[0][0].text)
        headings_array = np.array(output.get("high") + output.get("medium"))
        selected_heading = rng.choice(headings_array, size=1)[0]
        subheadings = headings[selected_heading]
        if subheadings:
            subheading = rng.choice(np.array(subheadings), size=1)[0]
            selected_heading = [selected_heading, subheading]

        return current_node[0], selected_heading
        nodes = [
            Node(**relation["target"])
            for relation in current_node[0].relationships
            if relation["label"] == "contains"
            and json.loads(relation["properties"]) in selected_heading
        ]

        return nodes

    def critic_question(self, question: str) -> bool:
        pass

    def retrieve_chunks(
        self, question: str, nodes: t.List[Node], kwargs: dict | None = None
    ) -> t.Any:
        pass

    def generate_answer(self, question: str, chunks: t.List[LCDocument]) -> t.Any:
        pass

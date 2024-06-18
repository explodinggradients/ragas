import json
import logging
import typing as t
from dataclasses import dataclass, field

import numpy as np
import tiktoken
from langchain.utils.math import cosine_similarity, cosine_similarity_top_k
from langchain_core.documents import Document as LCDocument

from ragas.llms.prompt import Prompt
from ragas.testsetv3.graph import Node, NodeLevel
from ragas.testsetv3.questions.base import QAC, QAGenerator
from ragas.testsetv3.questions.prompts import (
    abstract_question_from_theme,
    common_theme_from_summaries,
    common_topic_from_keyphrases,
    comparative_question,
    critic_question,
    question_answering,
)

logger = logging.getLogger(__name__)


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
        assert self.llm is not None, "LLM is not initialized"

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

        current_nodes = self.get_random_node(results)

        related_nodes = [
            rel.target
            for rel in current_nodes[0].relationships
            if rel.target.label.name == "DOC"
        ]

        if not related_nodes:
            return QAC()

        current_nodes.extend(related_nodes)
        summaries = [item.properties["metadata"]["summary"] for item in current_nodes]
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
        assert self.llm is not None, "LLM is not initialized"

        output = await self.llm.generate(critic_question.format(question=question))
        output = json.loads(output.generations[0][0].text)
        return all(score >= 2 for score in output.values())

    async def retrieve_chunks(
        self, question: str, nodes: t.List[Node], kwargs: t.Optional[dict] = None
    ) -> t.List[LCDocument] | None:
        assert self.embedding is not None, "Embedding is not initialized"
        assert self.llm is not None, "LLLM is not initialized"

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

        nodes = results
        output_documents = [
            LCDocument(
                page_content=node.properties["metadata"]["summary"],
                metadata={"source": node.properties["metadata"]["source"]},
            )
            for node in nodes
        ]

        # query to get all child nodes of nodes
        nodes = [
            relationship.target for node in nodes for relationship in node.relationships
        ]
        chunks = [
            LCDocument(
                page_content=node.properties["page_content"],
                metadata=node.properties["metadata"],
            )
            for node in nodes
        ]
        chunks_embeddings = [
            node.properties["metadata"]["page_content_embedding"] for node in nodes
        ]

        question_embedding = await self.embedding.embed_text(question)
        similarity_matrix = cosine_similarity([question_embedding], chunks_embeddings)
        most_similar = np.flip(np.argsort(similarity_matrix[0]))
        ranked_chunks = [chunks[i] for i in most_similar]
        # TODO: allow for different models
        model_name = "gpt-2"
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
        assert self.llm is not None, "LLM is not initialized"

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
            source {{
                id
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
        current_nodes = self.get_random_node(result_nodes)

        indices = np.flip(
            np.argsort(
                [rel.properties["score"] for rel in current_nodes[0].relationships]
            )
        )[:3]
        related_nodes = [current_nodes[0].relationships[i].target for i in indices]
        current_nodes.extend(related_nodes)

        summaries = [item.properties["metadata"]["summary"] for item in current_nodes]
        summaries = "\n".join(
            [f"{i+1}. {summary}" for i, summary in enumerate(summaries)]
        )
        common_theme = await self.llm.generate(
            self.generate_common_theme_prompt.format(summaries=summaries)
        )
        common_theme = common_theme.generations[0][0].text

        keyphrases = [
            node.properties["metadata"]["keyphrases"] for node in current_nodes
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

        kwargs = {"max_tokens": 4024, "common_theme": common_theme}

        critic_verdict = await self.critic_question(question)
        if critic_verdict:
            source = await self.retrieve_chunks(question, current_nodes, kwargs)
            question = await self.modify_question(question)
            if source:
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
                logger.warning("source not found: %s", question)
                return QAC()
        else:
            logger.warning("Critic rejected the question: %s", question)
            return QAC()

    async def critic_question(self, question: str) -> bool:
        assert self.llm is not None, "LLM is not initialized"

        output = await self.llm.generate(critic_question.format(question=question))
        output = json.loads(output.generations[0][0].text)
        return all(score >= 2 for score in output.values())

    async def generate_answer(self, question: str, chunks: t.List[LCDocument]) -> str:
        assert self.llm is not None, "LLM is not initialized"

        text = "\n\n".join([chunk.page_content for chunk in chunks])
        output = await self.llm.generate(
            self.generate_answer_prompt.format(question=question, text=text)
        )
        return output.generations[0][0].text

    async def retrieve_chunks(
        self, question: str, nodes: t.List[Node], kwargs: t.Optional[dict] = None
    ) -> t.List[LCDocument] | None:
        kwargs = kwargs or {}
        assert self.embedding is not None, "Embedding is not initialized"

        common_theme = kwargs.get("common_theme", "")
        query_emebdding = await self.embedding.embed_text(common_theme)

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
                level
            }}
            }}
        }}
        }}
        """
        kwargs = {"node_ids": node_ids}
        target_nodes = self.query_nodes(query, kwargs)
        if not target_nodes:
            return None

        target_nodes = [
            relation.target for node in target_nodes for relation in node.relationships
        ]
        target_nodes = [
            node for node in target_nodes if node.level == NodeLevel.LEVEL_1.name
        ]
        context_embedding = [
            node.properties["metadata"]["page_content_embedding"]
            for node in target_nodes
        ]
        idxs, _ = cosine_similarity_top_k([query_emebdding], context_embedding, top_k=2)
        target_nodes = [target_nodes[idx[1]] for idx in idxs]
        documents = [
            LCDocument(
                page_content=node.properties["page_content"],
                metadata=node.properties["metadata"],
            )
            for node in target_nodes
        ]
        return documents

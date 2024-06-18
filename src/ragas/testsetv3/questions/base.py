import logging
import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from langchain_core.documents import Document as LCDocument

from ragas.embeddings import BaseRagasEmbeddings, embedding_factory
from ragas.llms.base import BaseRagasLLM, llm_factory
from ragas.llms.prompt import Prompt
from ragas.testsetv3.graph import Node, Relationship
from ragas.testsetv3.graph import schema as myschema
from ragas.testsetv3.questions.prompts import question_modification
from ragas.testsetv3.utils import GraphConverter, rng

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
    style: t.Optional[QuestionStyle] = QuestionStyle.PERFECT_GRAMMAR
    length: t.Optional[QuestionLength] = QuestionLength.MEDIUM


@dataclass
class QAGenerator(ABC):
    nodes: t.List[Node]
    relationships: t.List[Relationship]

    llm: t.Optional[BaseRagasLLM] = None
    embedding: t.Optional[BaseRagasEmbeddings] = None
    style: QuestionStyle = QuestionStyle.PERFECT_GRAMMAR
    length: QuestionLength = QuestionLength.MEDIUM
    question_modification_prompt: Prompt = field(
        default_factory=lambda: question_modification
    )

    def __post_init__(self):
        self.llm = self.llm or llm_factory()
        self.embedding = self.embedding or embedding_factory()

    @abstractmethod
    async def generate_question(
        self, query: t.Optional[str] = None, kwargs: t.Optional[dict] = None
    ) -> t.Any:
        pass

    @abstractmethod
    async def critic_question(self, question: str) -> bool:
        pass

    @abstractmethod
    async def generate_answer(self, question: str, chunks: t.List[LCDocument]) -> t.Any:
        pass

    def retrieve_chunks(
        self, question: str, nodes: t.List[Node], kwargs: t.Optional[dict] = None
    ) -> t.Any:
        pass

    async def modify_question(self, question: str) -> str:
        assert self.llm is not None, "LLM is not initialized"
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

        return GraphConverter.convert(results.data["filterNodes"])

    def get_random_node(self, nodes) -> t.List[Node]:
        nodes = [node for node in nodes if node.relationships]
        nodes_weights = np.array([node.properties.get("chances", 0) for node in nodes])
        if all(nodes_weights == 0):
            nodes_weights = np.ones(len(nodes_weights))
        nodes_weights = nodes_weights / sum(nodes_weights)
        return rng.choice(np.array(nodes), p=nodes_weights, size=1).tolist()
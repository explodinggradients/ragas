import logging
import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from langchain_core.documents import Document as LCDocument
from ragas_experimental.testset.graph import Node, Relationship
from ragas_experimental.testset.graph import schema as myschema
from ragas_experimental.testset.questions.prompts import (
    EXAMPLES_FOR_QUESTION_MODIFICATION,
    question_modification,
)
from ragas_experimental.testset.utils import GraphConverter, rng

from ragas.embeddings import BaseRagasEmbeddings
from ragas.llms.base import BaseRagasLLM
from ragas.llms.prompt import Prompt

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
class StyleLengthDistribution:
    style_length_distribution: t.Dict[
        t.Tuple[QuestionStyle, QuestionLength], float
    ] = field(
        default_factory=lambda: {
            (QuestionStyle.PERFECT_GRAMMAR, QuestionLength.MEDIUM): 1.0
        }
    )

    def __post_init__(self):
        self.validate()

    def validate(self):
        total = sum(self.style_length_distribution.values())
        if not abs(total - 1.0) < 1e-6:
            raise ValueError("The distribution proportions must sum up to 1.0")

    def get_num_samples(
        self, total_samples: int, style: QuestionStyle, length: QuestionLength
    ) -> int:
        proportion = self.style_length_distribution.get((style, length), 0)
        return int(total_samples * proportion)

    def items(self):
        return list(self.style_length_distribution.items())

    def values(self):
        return list(self.style_length_distribution.values())

    def keys(self):
        return list(self.style_length_distribution.keys())


@dataclass
class QAGenerator(ABC):
    nodes: t.Optional[t.List[Node]] = None
    relationships: t.Optional[t.List[Relationship]] = None
    distribution: t.Optional[StyleLengthDistribution] = None
    llm: t.Optional[BaseRagasLLM] = None
    embedding: t.Optional[BaseRagasEmbeddings] = None
    question_modification_prompt: Prompt = field(
        default_factory=lambda: question_modification
    )

    @abstractmethod
    async def generate_question(
        self,
        nodes,
        style: QuestionStyle,
        length: QuestionLength,
        kwargs: t.Optional[dict] = None,
    ) -> t.Any:
        pass

    @abstractmethod
    async def critic_question(self, question: str) -> bool:
        pass

    @abstractmethod
    async def generate_answer(self, question: str, chunks: t.List[LCDocument]) -> t.Any:
        pass

    def retrieve_chunks(
        self, nodes: t.List[Node], kwargs: t.Optional[dict] = None
    ) -> t.Any:
        pass

    async def modify_question(
        self, question: str, style: QuestionStyle, length: QuestionLength
    ) -> str:
        assert self.llm is not None, "LLM is not initialized"
        examples = [
            example
            for example in EXAMPLES_FOR_QUESTION_MODIFICATION
            if example["style"] == style.value and example["length"] == length.value
        ]
        self.question_modification_prompt.examples.extend(examples)
        p_value = self.question_modification_prompt.format(
            question=question, style=style.value, length=length.value
        )
        self.question_modification_prompt.examples = []
        result = await self.llm.generate(prompt=p_value)
        modified_question = result.generations[0][0].text
        return modified_question

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
        results = list(results.data.values())[0]
        return GraphConverter.convert(results)

    def get_random_node(self, nodes) -> t.List[Node]:
        nodes = [node for node in nodes if node.relationships]
        nodes_weights = np.array([node.properties.get("chances", 0) for node in nodes])
        if all(nodes_weights == 0):
            nodes_weights = np.ones(len(nodes_weights))
        nodes_weights = nodes_weights / sum(nodes_weights)
        return rng.choice(np.array(nodes), p=nodes_weights, size=1).tolist()


DEFAULT_DISTRIBUTION = StyleLengthDistribution(
    {
        (QuestionStyle.PERFECT_GRAMMAR, QuestionLength.MEDIUM): 0.25,
        (QuestionStyle.POOR_GRAMMAR, QuestionLength.MEDIUM): 0.25,
        (QuestionStyle.WEB_SEARCH_LIKE, QuestionLength.MEDIUM): 0.25,
        (QuestionStyle.MISSPELLED, QuestionLength.MEDIUM): 0.25,
    }
)

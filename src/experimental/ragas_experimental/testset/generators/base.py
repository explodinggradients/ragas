import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass

from langchain_core.documents import Document

from ragas.embeddings import BaseRagasEmbeddings, embedding_factory
from ragas.llms.base import BaseRagasLLM, llm_factory


@dataclass
class QADistribution:
    question_types: t.List[t.Any]
    probabilities: t.List[float]

    def keys(self):
        return self.question_types

    def values(self):
        return self.probabilities

    def items(self):
        return zip(self.keys(), self.values())


@dataclass
class TestGenerator(ABC):
    llm: BaseRagasLLM = llm_factory()
    embedding: BaseRagasEmbeddings = embedding_factory()

    @abstractmethod
    def generate(
        self,
        docs: t.Sequence[Document],
        test_size: int,
        distribution: QADistribution,
    ) -> t.Any:
        ...

    def generate_with_langchain_docs(
        self,
        docs: t.Sequence[Document],
        test_size: int,
        distribution: QADistribution,
    ) -> t.Any:
        ...

    def generate_with_llamaindex_docs(
        self,
        docs: t.List[Document],
        test_size: int,
        distribution: QADistribution,
    ) -> t.Any:
        ...

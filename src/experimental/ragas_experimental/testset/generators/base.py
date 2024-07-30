import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import pandas as pd
from langchain_core.documents import Document
from llama_index.core.schema import Document as LlamaindexDocument
from ragas_experimental.testset.questions import QAC

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
class TestDataset:
    qac: t.List[QAC]

    def to_pandas(self):
        data = []
        for row in self.qac:
            data.append(row.to_dict())
        return pd.DataFrame(data)


@dataclass
class TestGenerator(ABC):
    llm: BaseRagasLLM = field(default_factory=llm_factory)
    embedding: BaseRagasEmbeddings = field(default_factory=embedding_factory)

    @abstractmethod
    def generate(
        self,
        docs: t.Sequence[Document],
        test_size: int,
        distribution: QADistribution,
    ) -> TestDataset:
        ...

    def generate_with_langchain_docs(
        self,
        docs: t.Sequence[Document],
        test_size: int,
        distribution: QADistribution,
    ) -> TestDataset:
        return self.generate(docs, test_size, distribution)

    def generate_with_llamaindex_docs(
        self,
        docs: t.Sequence[LlamaindexDocument],
        test_size: int,
        distribution: QADistribution,
    ) -> TestDataset:
        docs = [doc.to_langchain_format() for doc in docs]
        return self.generate(docs, test_size, distribution)

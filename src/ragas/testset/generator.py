from __future__ import annotations

import logging
import typing as t
from dataclasses import dataclass

import pandas as pd
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings

from ragas._analytics import TesetGenerationEvent, track
from ragas.embeddings import BaseRagasEmbeddings
from ragas.executor import Executor
from ragas.llms import BaseRagasLLM, LangchainLLMWrapper
from ragas.testset.docstore import Document, DocumentStore, InMemoryDocumentStore
from ragas.testset.evolutions import (
    ComplexEvolution,
    CurrentNodes,
    DataRow,
    multi_context,
    reasoning,
    simple,
)
from ragas.testset.filters import EvolutionFilter, NodeFilter, QuestionFilter

if t.TYPE_CHECKING:
    from llama_index.readers.schema import Document as LlamaindexDocument
    from langchain_core.documents import Document as LCDocument

logger = logging.getLogger(__name__)

Distributions = t.Dict[t.Any, float]
DEFAULT_DISTRIBUTION = {simple: 0.5, reasoning: 0.25, multi_context: 0.25}


@dataclass
class TestDataset:
    """
    TestDataset class
    """

    test_data: t.List[DataRow]

    def to_pandas(self) -> pd.DataFrame:
        data_samples = []
        for data in self.test_data:
            data_dict = dict(data)
            data_dict["episode_done"] = True
            data_samples.append(data_dict)

        return pd.DataFrame.from_records(data_samples)


@dataclass
class TestsetGenerator:
    generator_llm: BaseRagasLLM
    critic_llm: BaseRagasLLM
    embeddings: BaseRagasEmbeddings
    docstore: DocumentStore

    @classmethod
    def with_openai(
        cls,
        generator_llm: str = "gpt-3.5-turbo",
        critic_llm: str = "gpt-4",
        embeddings: str = "text-embedding-ada-002",
        docstore: t.Optional[DocumentStore] = None,
        chunk_size: int = 512,
    ) -> "TestsetGenerator":
        generator_llm_model = LangchainLLMWrapper(ChatOpenAI(model=generator_llm))
        critic_llm_model = LangchainLLMWrapper(ChatOpenAI(model=critic_llm))
        embeddings_model = OpenAIEmbeddings(model=embeddings)
        if docstore is None:
            from langchain.text_splitter import TokenTextSplitter

            splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
            docstore = InMemoryDocumentStore(
                splitter=splitter, embeddings=embeddings_model
            )
            return cls(
                generator_llm=generator_llm_model,
                critic_llm=critic_llm_model,
                embeddings=embeddings_model,
                docstore=docstore,
            )
        else:
            return cls(
                generator_llm=generator_llm_model,
                critic_llm=critic_llm_model,
                embeddings=embeddings_model,
                docstore=docstore,
            )

    # if you add any arguments to this function, make sure to add them to
    # generate_with_langchain_docs as well
    def generate_with_llamaindex_docs(
        self,
        documents: t.Sequence[LlamaindexDocument],
        test_size: int,
        distributions: Distributions = {},
        with_debugging_logs=False,
    ):
        # chunk documents and add to docstore
        self.docstore.add_documents(
            [Document.from_llamaindex_document(doc) for doc in documents]
        )

        return self.generate(
            test_size=test_size,
            distributions=distributions,
            with_debugging_logs=with_debugging_logs,
        )

    # if you add any arguments to this function, make sure to add them to
    # generate_with_langchain_docs as well
    def generate_with_langchain_docs(
        self,
        documents: t.Sequence[LCDocument],
        test_size: int,
        distributions: Distributions = {},
        with_debugging_logs=False,
    ):
        # chunk documents and add to docstore
        self.docstore.add_documents(
            [Document.from_langchain_document(doc) for doc in documents]
        )

        return self.generate(
            test_size=test_size,
            distributions=distributions,
            with_debugging_logs=with_debugging_logs,
        )

    def generate(
        self,
        test_size: int,
        distributions: Distributions = DEFAULT_DISTRIBUTION,
        with_debugging_logs=False,
    ):
        # init filters and evolutions
        for evolution in distributions:
            if evolution.generator_llm is None:
                evolution.generator_llm = self.generator_llm
            if evolution.docstore is None:
                evolution.docstore = self.docstore

            if evolution.question_filter is None:
                evolution.question_filter = QuestionFilter(llm=self.critic_llm)
            if evolution.node_filter is None:
                evolution.node_filter = NodeFilter(llm=self.critic_llm)

            if isinstance(evolution, ComplexEvolution):
                evolution.init_evolution()
                if evolution.evolution_filter is None:
                    evolution.evolution_filter = EvolutionFilter(llm=self.critic_llm)
        if with_debugging_logs:
            from ragas.utils import patch_logger

            patch_logger("ragas.testset.evolutions", logging.DEBUG)

        exec = Executor(
            desc="Generating",
            keep_progress_bar=True,
            raise_exceptions=True,
            is_async=True,
        )

        current_nodes = [
            CurrentNodes(root_node=n, nodes=[n])
            for n in self.docstore.get_random_nodes(k=test_size)
        ]
        for evolution, probability in distributions.items():
            for i in range(round(probability * test_size)):
                exec.submit(
                    evolution.aevolve,
                    current_nodes[i],
                    name=f"{evolution.__class__.__name__}-{i}",
                )

        try:
            test_data_rows = exec.results()
        except ValueError as e:
            raise e
        test_dataset = TestDataset(test_data=test_data_rows)
        track(
            TesetGenerationEvent(
                event_type="testset_generation",
                evolutions={
                    k.__class__.__name__.lower(): v for k, v in distributions.items()
                },
                num_rows=len(test_dataset.test_data),
            )
        )

        return test_dataset

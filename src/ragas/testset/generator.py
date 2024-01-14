import typing as t
from dataclasses import dataclass

from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from ragas.llms import BaseRagasLLM, LangchainLLMWrapper
from ragas.embeddings import BaseRagasEmbeddings
from ragas.testset.docstore import DocumentStore, Document
from ragas.testset.evolutions import SimpleEvolution

from llama_index.readers.schema import Document as LlamaindexDocument


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
    ) -> "TestsetGenerator":
        generator_llm_model = LangchainLLMWrapper(ChatOpenAI(model=generator_llm))
        critic_llm_model = LangchainLLMWrapper(ChatOpenAI(model=critic_llm))
        embeddings_model = OpenAIEmbeddings(model=embeddings)
        return cls(
            generator_llm=generator_llm_model,
            critic_llm=critic_llm_model,
            embeddings=embeddings_model,
        )

    def generate_with_llamaindex_docs(self, documents: t.Sequence[LlamaindexDocument]):
        # chunk documents and add to docstore
        self.docstore.add_documents(
            [Document.from_llamaindex_document(doc) for doc in documents]
        )
        # create evolutions and add to executor queue
        # run till completion - keep updating progress bar

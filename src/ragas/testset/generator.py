import typing as t
from dataclasses import dataclass

from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from ragas.llms import BaseRagasLLM, LangchainLLMWrapper
from ragas.embeddings import BaseRagasEmbeddings

from llama_index.readers.schema import Document as LlamaindexDocument


@dataclass
class TestsetGenerator:
    generator_llm: BaseRagasLLM
    critic_llm: BaseRagasLLM
    embeddings: BaseRagasEmbeddings

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
        print(len(documents))
        # chunk documents and add to docstore
        # create evolutions and add to executor queue
        # run till completion - keep updating progress bar

import typing as t
from dataclasses import dataclass

from langchain.embeddings import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from llama_index.readers.schema import Document as LlamaindexDocument

from ragas.embeddings import BaseRagasEmbeddings
from ragas.llms import BaseRagasLLM, LangchainLLMWrapper
from ragas.testset.docstore import Document, DocumentStore, InMemoryDocumentStore
from ragas.executor import Executor
from ragas.testset.evolutions import SimpleEvolution, QuestionFilter, NodeFilter


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
            docstore = InMemoryDocumentStore(splitter)
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

    def generate_with_llamaindex_docs(self, documents: t.Sequence[LlamaindexDocument]):
        # chunk documents and add to docstore
        self.docstore.add_documents(
            [Document.from_llamaindex_document(doc) for doc in documents]
        )
        # create evolutions and add to executor queue
        # run till completion - keep updating progress bar
        #

    def generate(self, test_size: int):
        node_filter = NodeFilter(self.critic_llm)
        ques_filter = QuestionFilter(self.critic_llm)
        exec = Executor()
        qs = []
        for i in range(test_size):
            se = SimpleEvolution(node_filter, ques_filter)
            exec.submit(
                se.aevolve,
                self.generator_llm,
                self.docstore,
                name=f"SimpleEvolution-{i}",
            )
            try:
                qs = exec.results()
            except ValueError as e:
                raise e
        return qs

import time

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from llama_index.core import download_loader

from ragas.testset.evolutions import conditional, multi_context, reasoning, simple
from ragas.testset.generator import TestsetGenerator

generator_llm = ChatOpenAI(model="gpt-3.5-turbo-16k")
critic_llm = ChatOpenAI(model="gpt-4")
embeddings = OpenAIEmbeddings()

generator = TestsetGenerator.from_langchain(generator_llm, critic_llm, embeddings)

distributions = {simple: 0.5, multi_context: 0.3, reasoning: 0.1, conditional: 0.1}


def get_documents():
    SemanticScholarReader = download_loader("SemanticScholarReader")
    loader = SemanticScholarReader()
    # Narrow down the search space
    query_space = "large language models"
    # Increase the limit to obtain more documents
    documents = loader.load_data(query=query_space, limit=10)

    return documents


IGNORE_ASYNCIO = False
# os.environ["PYTHONASYNCIODEBUG"] = "1"

if __name__ == "__main__":
    documents = get_documents()

    # asyncio
    print("Starting [Asyncio]")
    start = time.time()
    generator.generate_with_llamaindex_docs(
        documents=documents,
        test_size=50,
        distributions=distributions,
        is_async=True,
    )
    print(f"Time taken: {time.time() - start:.2f}s")

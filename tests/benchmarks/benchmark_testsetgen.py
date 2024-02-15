import time

from llama_index import download_loader

from ragas.testset.evolutions import conditional, multi_context, reasoning, simple
from ragas.testset.generator import TestsetGenerator

generator = TestsetGenerator.with_openai()

distributions = {simple: 0.5, multi_context: 0.3, reasoning: 0.1, conditional: 0.1}


def get_documents():
    SemanticScholarReader = download_loader("SemanticScholarReader")
    loader = SemanticScholarReader()
    # Narrow down the search space
    query_space = "large language models"
    # Increase the limit to obtain more documents
    documents = loader.load_data(query=query_space, limit=10)

    return documents


IGNORE_THREADS = False
IGNORE_ASYNCIO = False
# os.environ["PYTHONASYNCIODEBUG"] = "1"

if __name__ == "__main__":
    documents = get_documents()

    # asyncio
    if not IGNORE_ASYNCIO:
        print("Starting [Asyncio]")
        start = time.time()
        generator.generate_with_llamaindex_docs(
            documents=documents,
            test_size=50,
            distributions=distributions,
            is_async=True,
        )
        print(f"Time taken: {time.time() - start:.2f}s")

    # Threads
    if not IGNORE_THREADS:
        print("Starting [Threads]")
        start = time.time()
        generator.generate_with_llamaindex_docs(
            documents=documents,
            test_size=50,
            distributions=distributions,
            is_async=False,
        )
        print(f"Time taken [Threads]: {time.time() - start:.2f}s")

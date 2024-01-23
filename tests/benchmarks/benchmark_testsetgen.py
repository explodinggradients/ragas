import os
import time

from llama_index import download_loader

from ragas.testset.evolutions import multi_context, reasoning, simple
from ragas.testset.generator import TestsetGenerator

generator = TestsetGenerator.with_openai()

distributions = {simple: 0.5, multi_context: 0.4, reasoning: 0.1}


def get_documents():
    SemanticScholarReader = download_loader("SemanticScholarReader")
    loader = SemanticScholarReader()
    # Narrow down the search space
    query_space = "large language models"
    # Increase the limit to obtain more documents
    documents = loader.load_data(query=query_space, limit=10)

    return documents


IGNORE_THREADS = True
IGNORE_ASYNCIO = False

if __name__ == "__main__":
    documents = get_documents()

    # asyncio
    if not IGNORE_ASYNCIO:
        os.environ["PYTHONASYNCIODEBUG"] = "1"
        print("Starting [Asyncio]")
        start = time.time()
        generator.generate_with_llamaindex_docs(
            documents=documents, test_size=100, distributions=distributions
        )
        print(f"Time taken: {time.time() - start:.2f}s")

    # Threads
    if not IGNORE_THREADS:
        print("Starting [Threads]")

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from llama_index.core import download_loader

from ragas.testset.synthesizers.generate import TestsetGenerator

generator_llm = ChatOpenAI(model="gpt-4o")
embeddings = OpenAIEmbeddings()

generator = TestsetGenerator.from_langchain(generator_llm, embeddings)


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
    generator.generate_with_llamaindex_docs(
        documents=documents,
        testset_size=50,
    )

import os

import pytest

from ragas.testset import TestsetGenerator


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_testset_generation_e2e():
    # generate kg
    from langchain_community.document_loaders import DirectoryLoader

    loader = DirectoryLoader("./docs", glob="**/*.md")
    docs = loader.load()

    # choose llm
    from ragas.embeddings import embedding_factory
    from ragas.llms import llm_factory

    generator_llm = llm_factory("gpt-4o")
    generator_embeddings = embedding_factory()

    generator = TestsetGenerator(
        llm=generator_llm,
        embedding_model=generator_embeddings,  # type: ignore
    )
    dataset = generator.generate_with_langchain_docs(docs, testset_size=3)
    assert dataset is not None

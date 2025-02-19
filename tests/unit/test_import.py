from __future__ import annotations

import builtins
from unittest.mock import MagicMock

import pytest


def test_missing_haystack_llmwrapper(monkeypatch):
    real_import = builtins.__import__

    def mocked_import(name, *args, **kwargs):
        if name.startswith("haystack"):
            raise ImportError("No module named 'haystack'")
        return real_import(name, *args, **kwargs)

    # Replace the built-in import function with our mock
    monkeypatch.setattr(builtins, "__import__", mocked_import)

    # Test: Non-Haystack wrappers still work fine
    from langchain_openai.llms import OpenAI

    from ragas.llms import LangchainLLMWrapper

    langchain_mocked_llm = MagicMock(spec=OpenAI)
    langchain_mocked_llm.model_name = "gpt-3.5-turbo-instruct"

    langchain_wrapper = LangchainLLMWrapper(langchain_llm=langchain_mocked_llm)

    assert langchain_wrapper.langchain_llm.model_name == "gpt-3.5-turbo-instruct"

    # Test: Importing HaystackLLMWrapper fails
    with pytest.raises(ImportError, match="Haystack is not installed"):
        from ragas.llms import HaystackLLMWrapper

        HaystackLLMWrapper(haystack_generator=None)


def test_wrappers_with_missing_haystack(monkeypatch):
    """Simulate missing 'haystack' and verify that:
    - Non-Haystack wrappers import and instantiate without error.
    - Importing HaystackEmbeddingsWrapper fails with an ImportError.
    """

    real_import = builtins.__import__

    # Define our mock import function that raises ImportError if "haystack" is imported
    def mocked_import(name, *args, **kwargs):
        if name.startswith("haystack"):
            raise ImportError("No module named 'haystack'")
        return real_import(name, *args, **kwargs)

    # Replace the built-in import with our mock
    monkeypatch.setattr(builtins, "__import__", mocked_import)

    # Test: Non-Haystack wrappers still work fine
    from langchain_openai.embeddings import OpenAIEmbeddings
    from llama_index.core.base.embeddings.base import BaseEmbedding

    from ragas.embeddings import LangchainEmbeddingsWrapper, LlamaIndexEmbeddingsWrapper

    langchain_mocked_embedding = MagicMock(spec=OpenAIEmbeddings)
    langchain_mocked_embedding.model = "text-embedding-ada-002"
    llama_index_mocked_embedding = MagicMock(spec=BaseEmbedding)

    langchain_wrapper = LangchainEmbeddingsWrapper(
        embeddings=langchain_mocked_embedding
    )
    llama_index_wrapper = LlamaIndexEmbeddingsWrapper(
        embeddings=llama_index_mocked_embedding
    )

    assert langchain_wrapper.embeddings.model == "text-embedding-ada-002"
    assert llama_index_wrapper.embeddings is llama_index_mocked_embedding

    # Test: Importing HaystackEmbeddingsWrapper fails
    with pytest.raises(ImportError, match="Haystack is not installed"):
        from ragas.embeddings import HaystackEmbeddingsWrapper

        HaystackEmbeddingsWrapper(embedder=None)


def test_import_module():
    import ragas.metrics
    import ragas.metrics._aspect_critic

    test_metrics = [
        "answer_correctness",
        "answer_relevancy",
        "answer_similarity",
        "context_recall",
        "context_precision",
        "faithfulness",
    ]

    aspect_critics = [
        "harmfulness",
        "maliciousness",
        "coherence",
        "correctness",
        "conciseness",
    ]

    assert ragas.metrics is not None, "module is not imported"

    for metric in test_metrics:
        assert hasattr(ragas.metrics, metric)

    for metric in aspect_critics:
        assert hasattr(ragas.metrics._aspect_critic, metric)


def test_import_in_debug_mode():
    """
    if `RAGAS_DEBUG` is set to `True`, the module should be imported with
    logging level set to `DEBUG`
    """
    import os

    from ragas.utils import get_debug_mode

    get_debug_mode.cache_clear()

    os.environ["RAGAS_DEBUG"] = "True"

    assert get_debug_mode() is True

    del os.environ["RAGAS_DEBUG"]
    get_debug_mode.cache_clear()

from __future__ import annotations

import builtins

import pytest


def test_missing_haystack_llmwrapper(monkeypatch):
    real_import = builtins.__import__

    def mocked_import(name, *args, **kwargs):
        if name.startswith("haystack"):
            raise ImportError("No module named 'haystack'")
        return real_import(name, *args, **kwargs)

    # Replace the built-in import function with our mock
    monkeypatch.setattr(builtins, "__import__", mocked_import)

    with pytest.raises(ImportError, match="Haystack is not installed"):
        from ragas.llms import HaystackLLMWrapper

        HaystackLLMWrapper(haystack_generator=None)


def test_missing_haystack_embeddingswrapper(monkeypatch):
    # Save a reference to the real built-in import
    real_import = builtins.__import__

    def mocked_import(name, *args, **kwargs):
        # If we're trying to import "haystack", simulate it's missing
        if name.startswith("haystack"):
            raise ImportError("No module named 'haystack'")
        return real_import(name, *args, **kwargs)

    # Replace the built-in import function with our mock
    monkeypatch.setattr(builtins, "__import__", mocked_import)

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

from __future__ import annotations

import builtins
import typing as t

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


def test_import():
    import ragas
    from ragas.testset import TestsetGenerator

    assert TestsetGenerator is not None
    assert ragas is not None


def test_type_casting():
    t.cast(t.List[int], [1, 2, 3])


def test_import_metrics():
    from ragas.metrics._aspect_critic import harmfulness

    assert harmfulness is not None

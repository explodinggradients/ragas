from __future__ import annotations

import typing as t

import numpy as np
import pytest


def test_basic_imports():
    """Test that basic imports work."""
    from ragas.embeddings import BaseRagasEmbeddings, embedding_factory
    assert BaseRagasEmbeddings is not None
    assert embedding_factory is not None


def test_modern_imports():
    """Test that modern imports work."""
    from ragas.embeddings import (
        RagasBaseEmbedding,
        modern_embedding_factory,
        OpenAIEmbeddings,
        GoogleEmbeddings,
        LiteLLMEmbeddings,
        HuggingFaceEmbeddings,
    )
    assert RagasBaseEmbedding is not None
    assert modern_embedding_factory is not None
    assert OpenAIEmbeddings is not None
    assert GoogleEmbeddings is not None
    assert LiteLLMEmbeddings is not None
    assert HuggingFaceEmbeddings is not None


def test_utilities():
    """Test embedding utilities."""
    from ragas.embeddings import validate_texts, batch_texts, get_optimal_batch_size
    
    # Test validate_texts
    texts = validate_texts(["text1", "text2"])
    assert texts == ["text1", "text2"]
    
    # Test batch_texts
    batches = batch_texts(["a", "b", "c", "d"], 2)
    assert batches == [["a", "b"], ["c", "d"]]
    
    # Test get_optimal_batch_size
    size = get_optimal_batch_size("openai", "text-embedding-3-small")
    assert size == 100


def test_embedding_factory_legacy():
    """Test legacy embedding factory."""
    from ragas.embeddings import embedding_factory
    
    # Test that it raises appropriate error for modern interface
    with pytest.raises(ValueError, match="Modern interface requires"):
        embedding_factory(interface="modern")


def test_modern_embedding_errors():
    """Test modern embedding factory error handling."""
    from ragas.embeddings import modern_embedding_factory
    
    # Test provider validation
    with pytest.raises(ValueError, match="Unsupported provider"):
        modern_embedding_factory("invalid_provider")
    
    # Test OpenAI requires client
    with pytest.raises(ValueError, match="OpenAI provider requires a client"):
        modern_embedding_factory("openai", "text-embedding-3-small")
    
    # Test Google requires client
    with pytest.raises(ValueError, match="Google provider requires a client"):
        modern_embedding_factory("google", "text-embedding-004")
    
    # Test LiteLLM requires model
    with pytest.raises(ValueError, match="LiteLLM provider requires a model"):
        modern_embedding_factory("litellm")
    
    # Test HuggingFace requires model
    with pytest.raises(ValueError, match="HuggingFace provider requires a model"):
        modern_embedding_factory("huggingface")


class MockEmbedding:
    """Simple mock embedding for basic testing."""

    def __init__(self):
        from ragas.embeddings import RagasBaseEmbedding
        self.base_class = RagasBaseEmbedding

    def embed_text(self, text: str, **kwargs: t.Any) -> t.List[float]:
        np.random.seed(42)  # Set seed for deterministic tests
        return np.random.rand(768).tolist()

    async def aembed_text(self, text: str, **kwargs: t.Any) -> t.List[float]:
        np.random.seed(42)  # Set seed for deterministic tests
        return np.random.rand(768).tolist()


def test_mock_embedding():
    """Test that mock embedding works correctly."""
    embedding = MockEmbedding()
    
    # Test sync embedding
    result = embedding.embed_text("test text")
    assert isinstance(result, list)
    assert len(result) == 768
    assert all(isinstance(x, float) for x in result)


@pytest.mark.asyncio
async def test_mock_embedding_async():
    """Test that mock embedding works correctly in async mode."""
    embedding = MockEmbedding()
    
    # Test async embedding
    result = await embedding.aembed_text("test text")
    assert isinstance(result, list)
    assert len(result) == 768
    assert all(isinstance(x, float) for x in result)
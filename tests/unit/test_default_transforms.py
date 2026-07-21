import pytest
from langchain_core.documents import Document as LCDocument

from ragas.testset.transforms.default import default_transforms


class FakeLLM:
    pass


class FakeEmbeddingModel:
    pass


def test_default_transforms_empty_documents_raises_clear_error():
    """An empty documents list must fail with a clear, actionable error instead of
    an opaque ZeroDivisionError from the internal token-length-bin computation."""
    with pytest.raises(ValueError, match="documents"):
        default_transforms(
            documents=[], llm=FakeLLM(), embedding_model=FakeEmbeddingModel()
        )


def test_default_transforms_too_short_documents_raises_value_error():
    documents = [LCDocument(page_content="short doc")]
    with pytest.raises(ValueError, match="too short"):
        default_transforms(
            documents=documents, llm=FakeLLM(), embedding_model=FakeEmbeddingModel()
        )


def test_default_transforms_with_long_documents_returns_transforms():
    documents = [LCDocument(page_content="word " * 600)]
    transforms = default_transforms(
        documents=documents, llm=FakeLLM(), embedding_model=FakeEmbeddingModel()
    )
    assert len(transforms) > 0

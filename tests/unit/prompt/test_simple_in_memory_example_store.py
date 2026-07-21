import typing as t

from ragas.embeddings.base import BaseRagasEmbedding as BaseEmbedding
from ragas.prompt.dynamic_few_shot import SimpleInMemoryExampleStore


class ConstantEmbeddingModel(BaseEmbedding):
    """Returns the same embedding for every input, so every stored example is
    equally similar to any query and none are excluded by the threshold."""

    def _generate_embedding(self, text: str) -> t.List[float]:
        return [1.0, 0.0, 0.0]

    def embed_text(self, text: str, **kwargs: t.Any) -> t.List[float]:
        return self._generate_embedding(text)

    async def aembed_text(self, text: str, **kwargs: t.Any) -> t.List[float]:
        return self._generate_embedding(text)

    def embed_query(self, text: str) -> t.List[float]:
        return self._generate_embedding(text)

    async def aembed_query(self, text: str) -> t.List[float]:
        return self._generate_embedding(text)


def _populated_store(embedding_model=None) -> SimpleInMemoryExampleStore:
    store = SimpleInMemoryExampleStore(embedding_model=embedding_model)
    for i in range(5):
        store.add_example({"q": f"question {i}"}, {"a": f"answer {i}"})
    return store


def test_get_examples_top_k_zero_returns_none_without_embedding_model():
    """top_k=0 must mean 'no examples', not 'all examples' -- Python's
    list[-0:] slicing quirk (-0 == 0) makes this an easy trap in the
    recency-based fallback path used when no embedding model is set."""
    store = _populated_store()
    result = store.get_examples({"q": "new question"}, top_k=0)
    assert result == []


def test_get_examples_top_k_zero_returns_none_with_embedding_model():
    """Same top_k=0 contract, but through the cosine-similarity path used
    when an embedding model is configured."""
    store = _populated_store(embedding_model=ConstantEmbeddingModel())
    result = store.get_examples({"q": "new question"}, top_k=0, threshold=0.0)
    assert result == []


def test_get_examples_top_k_positive_respects_limit_without_embedding_model():
    store = _populated_store()
    result = store.get_examples({"q": "new question"}, top_k=2)
    assert len(result) == 2


def test_get_examples_top_k_positive_respects_limit_with_embedding_model():
    store = _populated_store(embedding_model=ConstantEmbeddingModel())
    result = store.get_examples({"q": "new question"}, top_k=2, threshold=0.0)
    assert len(result) == 2

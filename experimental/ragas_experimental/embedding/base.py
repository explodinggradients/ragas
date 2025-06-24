__all__ = ["BaseEmbedding", "OpenAIEmbeddings", "ragas_embedding"]

import typing as t
from abc import ABC, abstractmethod


# TODO: Add support for other providers like HuggingFace, Cohere, etc.
# TODO: handle async calls properly and ensure that the client supports async if needed.
class BaseEmbedding(ABC):
    @abstractmethod
    def embed_text(self, text: str, **kwargs: t.Any) -> t.List[float]:
        pass

    @abstractmethod
    async def aembed_text(self, text: str, **kwargs: t.Any) -> t.List[float]:
        pass

    @abstractmethod
    def embed_document(
        self, documents: t.List[str], **kwargs: t.Any
    ) -> t.List[t.List[float]]:
        pass

    @abstractmethod
    async def aembed_document(
        self, documents: t.List[str], **kwargs: t.Any
    ) -> t.List[t.List[float]]:
        pass


class OpenAIEmbeddings(BaseEmbedding):
    def __init__(self, client: t.Any, model: str):
        self.client = client
        self.model = model

    def embed_text(self, text: str, **kwargs: t.Any) -> t.List[float]:
        return (
            self.client.embeddings.create(input=text, model=self.model, **kwargs)
            .data[0]
            .embedding
        )

    async def aembed_text(self, text: str, **kwargs: t.Any) -> t.List[float]:
        response = await self.client.embeddings.create(
            input=text, model=self.model, **kwargs
        )
        return response.data[0].embedding

    def embed_document(
        self, documents: t.List[str], **kwargs: t.Any
    ) -> t.List[t.List[float]]:
        embeddings = self.client.embeddings.create(
            input=documents, model=self.model, **kwargs
        )
        return [embedding.embedding for embedding in embeddings.data]

    async def aembed_document(
        self, documents: t.List[str], **kwargs: t.Any
    ) -> t.List[t.List[float]]:
        embeddings = await self.client.embeddings.create(
            input=documents, model=self.model, **kwargs
        )
        return [embedding.embedding for embedding in embeddings.data]


def ragas_embedding(provider: str, model: str, client: t.Any) -> BaseEmbedding:
    """
    Factory function to create an embedding instance based on the provider.

    Args:
        provider (str): The name of the embedding provider (e.g., "openai").
        model (str): The model name to use for embeddings.
        **kwargs: Additional arguments for the provider's client.

    Returns:
        BaseEmbedding: An instance of the specified embedding provider.
    """
    if provider.lower() == "openai":
        return OpenAIEmbeddings(client=client, model=model)

    raise ValueError(f"Unsupported provider: {provider}")

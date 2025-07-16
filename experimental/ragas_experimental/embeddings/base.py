__all__ = ["BaseEmbedding", "OpenAIEmbeddings", "embedding_factory"]

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

    def embed_texts(self, texts: t.List[str], **kwargs: t.Any) -> t.List[t.List[float]]:
        return [self.embed_text(text, **kwargs) for text in texts]

    async def aembed_texts(
        self, texts: t.List[str], **kwargs: t.Any
    ) -> t.List[t.List[float]]:
        return [await self.aembed_text(text, **kwargs) for text in texts]


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


def embedding_factory(provider: str, model: str, client: t.Any) -> BaseEmbedding:
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

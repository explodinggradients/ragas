__all__ = ["BaseEmbedding", "embedding_factory"]

import asyncio
import inspect
import typing as t
from abc import ABC, abstractmethod

from .utils import run_async_in_current_loop, validate_texts


class BaseEmbedding(ABC):
    """Abstract base class for embedding implementations.

    This class provides a consistent interface for embedding text using various
    providers. Implementations should provide both sync and async methods for
    embedding single texts, with batch methods automatically provided.
    """

    @abstractmethod
    def embed_text(self, text: str, **kwargs: t.Any) -> t.List[float]:
        """Embed a single text.

        Args:
            text: The text to embed
            **kwargs: Additional arguments for the embedding call

        Returns:
            List of floats representing the embedding
        """
        pass

    @abstractmethod
    async def aembed_text(self, text: str, **kwargs: t.Any) -> t.List[float]:
        """Asynchronously embed a single text.

        Args:
            text: The text to embed
            **kwargs: Additional arguments for the embedding call

        Returns:
            List of floats representing the embedding
        """
        pass

    def embed_texts(self, texts: t.List[str], **kwargs: t.Any) -> t.List[t.List[float]]:
        """Embed multiple texts.

        Default implementation processes texts individually. Override for
        batch optimization.

        Args:
            texts: List of texts to embed
            **kwargs: Additional arguments for the embedding calls

        Returns:
            List of embeddings, one for each input text
        """
        texts = validate_texts(texts)
        return [self.embed_text(text, **kwargs) for text in texts]

    async def aembed_texts(
        self, texts: t.List[str], **kwargs: t.Any
    ) -> t.List[t.List[float]]:
        """Asynchronously embed multiple texts.

        Default implementation processes texts concurrently. Override for
        batch optimization.

        Args:
            texts: List of texts to embed
            **kwargs: Additional arguments for the embedding calls

        Returns:
            List of embeddings, one for each input text
        """
        texts = validate_texts(texts)
        tasks = [self.aembed_text(text, **kwargs) for text in texts]
        return await asyncio.gather(*tasks)

    def _check_client_async(
        self, client: t.Any, method_path: str = "embeddings.create"
    ) -> bool:
        """Check if a client supports async operations.

        Args:
            client: The client to check
            method_path: Dot-separated path to the method to check

        Returns:
            True if the client supports async operations
        """
        try:
            obj = client
            for attr in method_path.split("."):
                obj = getattr(obj, attr)
            return inspect.iscoroutinefunction(obj)
        except (AttributeError, TypeError):
            return False

    def _run_async_in_current_loop(self, coro):
        """Run an async coroutine in the current event loop if possible.

        This handles Jupyter environments correctly by using a separate thread
        when a running event loop is detected.

        Args:
            coro: The coroutine to run

        Returns:
            The result of the coroutine
        """
        return run_async_in_current_loop(coro)


def embedding_factory(
    provider: str,
    model: t.Optional[str] = None,
    client: t.Optional[t.Any] = None,
    **kwargs: t.Any,
) -> BaseEmbedding:
    """
    Factory function to create an embedding instance based on the provider.

    Args:
        provider (str): The name of the embedding provider or provider/model string
                       (e.g., "openai", "openai/text-embedding-3-small").
        model (str, optional): The model name to use for embeddings.
        client (Any, optional): Pre-initialized client for the provider.
        **kwargs: Additional arguments for the provider.

    Returns:
        BaseEmbedding: An instance of the specified embedding provider.

    Examples:
        # OpenAI with client
        embedder = embedding_factory("openai", "text-embedding-3-small", client=openai_client)

        # OpenAI with provider/model string
        embedder = embedding_factory("openai/text-embedding-3-small", client=openai_client)

        # Google with Vertex AI
        embedder = embedding_factory(
            "google",
            "text-embedding-004",
            client=vertex_client,
            use_vertex=True,
            project_id="my-project"
        )

        # LiteLLM (supports 100+ models)
        embedder = embedding_factory("litellm", "text-embedding-ada-002", api_key="sk-...")

        # HuggingFace local model
        embedder = embedding_factory("huggingface", "sentence-transformers/all-MiniLM-L6-v2")
    """
    # Handle provider/model string format
    if "/" in provider and model is None:
        provider_name, model_name = provider.split("/", 1)
        provider = provider_name
        model = model_name

    provider_lower = provider.lower()

    if provider_lower == "openai":
        if not client:
            raise ValueError("OpenAI provider requires a client instance")
        from .openai import OpenAIEmbeddings

        return OpenAIEmbeddings(client=client, model=model or "text-embedding-3-small")

    elif provider_lower == "google":
        if not client:
            raise ValueError("Google provider requires a client instance")
        from .google import GoogleEmbeddings

        return GoogleEmbeddings(
            client=client, model=model or "text-embedding-004", **kwargs
        )

    elif provider_lower == "litellm":
        if not model:
            raise ValueError("LiteLLM provider requires a model name")
        from .litellm import LiteLLMEmbeddings

        return LiteLLMEmbeddings(model=model, **kwargs)

    elif provider_lower == "huggingface":
        if not model:
            raise ValueError("HuggingFace provider requires a model name")
        from .huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model=model, **kwargs)

    else:
        raise ValueError(
            f"Unsupported provider: {provider}. "
            f"Supported providers: openai, google, litellm, huggingface"
        )

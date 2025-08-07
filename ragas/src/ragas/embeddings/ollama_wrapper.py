import typing as t
from langchain_ollama import OllamaEmbeddings

from ragas.cache import CacheInterface
from ragas.embeddings.base import BaseRagasEmbeddings
from ragas.run_config import RunConfig


class OllamaEmbeddingsWrapper(BaseRagasEmbeddings):
    """
    A wrapper class for using Ollama Embeddings within the Ragas framework.

    This class integrates Ollama's Embeddings into Ragas, enabling both synchronous and
    asynchronous embedding generation.

    Parameters
    ----------
    ollama_embeddings : OllamaEmbeddings
        An instance of Ollama embeddings model.
    run_config : RunConfig, optional
        Configuration object to manage embedding execution settings, by default None.
    cache : CacheInterface, optional
        A cache instance for storing results, by default None.
    """

    def __init__(
        self,
        ollama_embeddings: OllamaEmbeddings,
        run_config: t.Optional[RunConfig] = None,
        cache: t.Optional[CacheInterface] = None,
    ):
        super().__init__(cache=cache)
        self.embeddings = ollama_embeddings

        if run_config is None:
            run_config = RunConfig()
        self.set_run_config(run_config)

    def embed_query(self, text: str) -> t.List[float]:
        """Generate embedding for a single text."""
        return self.embeddings.embed_query(text)

    def embed_documents(self, texts: t.List[str]) -> t.List[t.List[float]]:
        """Generate embeddings for multiple texts."""
        return self.embeddings.embed_documents(texts)

    async def aembed_query(self, text: str) -> t.List[float]:
        """Generate embedding for a single text asynchronously."""
        return await self.embeddings.aembed_query(text)

    async def aembed_documents(self, texts: t.List[str]) -> t.List[t.List[float]]:
        """Generate embeddings for multiple texts asynchronously."""
        return await self.embeddings.aembed_documents(texts)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(embeddings={self.embeddings.model}(...))" 
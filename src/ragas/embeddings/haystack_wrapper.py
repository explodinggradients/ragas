import asyncio
import typing as t

try:
    from haystack_experimental.core import AsyncPipeline
except ImportError:
    raise ImportError(
        "haystack-experimental is not installed. Please install it using `pip install haystack-experimental==0.4.0`."
    )
try:
    from haystack.components.embedders import (  # type: ignore
        AzureOpenAITextEmbedder,
        HuggingFaceAPITextEmbedder,
        OpenAITextEmbedder,
        SentenceTransformersTextEmbedder,
    )
except ImportError:
    raise ImportError(
        "pip install haystack-ai is not installed. Please install it using `pip install pip install haystack-ai`."
    )


from ragas.cache import CacheInterface
from ragas.embeddings.base import BaseRagasEmbeddings
from ragas.run_config import RunConfig


class HaystackEmbeddingsWrapper(BaseRagasEmbeddings):
    def __init__(
        self,
        embedder: t.Union[
            OpenAITextEmbedder,  # type: ignore
            SentenceTransformersTextEmbedder,  # type: ignore
            HuggingFaceAPITextEmbedder,  # type: ignore
            AzureOpenAITextEmbedder,  # type: ignore
        ],
        run_config: t.Optional[RunConfig] = None,
        cache: t.Optional[CacheInterface] = None,
    ):
        super().__init__(cache=cache)
        self.embedder = embedder
        self.async_pipeline = AsyncPipeline()
        self.async_pipeline.add_component("embedder", self.embedder)
        if run_config is None:
            run_config = RunConfig()
        self.set_run_config(run_config)

    def embed_query(self, text: str) -> t.List[float]:
        """
        Embed a single query text.
        """
        return self.embedder.run(text=text)["embedding"]

    def embed_documents(self, texts: t.List[str]) -> t.List[t.List[float]]:
        """
        Embed multiple documents.
        """
        return [self.embed_query(text) for text in texts]

    async def aembed_query(self, text: str) -> t.List[float]:
        """
        Asynchronously embed a single query text.
        """

        async def embedding_pipeline(text: str):
            result = []

            async for output in self.async_pipeline.run({"embedder": {"text": text}}):
                if "embedder" in output and "embedding" in output["embedder"]:
                    result = output["embedder"]["embedding"]
                    break

            return result

        return await embedding_pipeline(text=text)

    async def aembed_documents(self, texts: t.List[str]) -> t.List[t.List[float]]:
        """
        Asynchronously embed multiple documents.
        """
        results = await asyncio.gather(*(self.aembed_query(text) for text in texts))
        return results

    def __repr__(self) -> str:
        if isinstance(
            self.embedder, (OpenAITextEmbedder, SentenceTransformersTextEmbedder)  # type: ignore
        ):
            model = self.embedder.model
        elif isinstance(self.embedder, AzureOpenAITextEmbedder):  # type: ignore
            model = self.embedder.azure_deployment
        elif isinstance(self.embedder, HuggingFaceAPITextEmbedder):  # type: ignore
            model = self.embedder.api_params
        else:
            model = "Unknown"

        return f"{self.__class__.__name__}(embeddings={model}(...))"

"""Google embeddings implementation supporting both Vertex AI and Google AI (Gemini)."""

import typing as t

from .base import BaseRagasEmbedding
from .utils import run_sync_in_async, validate_texts


class GoogleEmbeddings(BaseRagasEmbedding):
    """Google embeddings using Vertex AI or Google AI (Gemini).

    Supports both Vertex AI and Google AI (Gemini) embedding models.
    For Vertex AI, requires google-cloud-aiplatform package.
    For Google AI, requires google-generativeai package.
    """

    PROVIDER_NAME = "google"
    REQUIRES_CLIENT = True
    DEFAULT_MODEL = "text-embedding-004"

    def __init__(
        self,
        client: t.Any,
        model: str = "text-embedding-004",
        use_vertex: bool = False,
        project_id: t.Optional[str] = None,
        location: t.Optional[str] = "us-central1",
        **kwargs: t.Any,
    ):
        self.client = client
        self.model = model
        self.use_vertex = use_vertex
        self.project_id = project_id
        self.location = location
        self.kwargs = kwargs

    def embed_text(self, text: str, **kwargs: t.Any) -> t.List[float]:
        """Embed a single text using Google's embedding service."""
        if self.use_vertex:
            return self._embed_text_vertex(text, **kwargs)
        else:
            return self._embed_text_genai(text, **kwargs)

    def _embed_text_vertex(self, text: str, **kwargs: t.Any) -> t.List[float]:
        """Embed text using Vertex AI."""
        try:
            from vertexai.language_models import TextEmbeddingModel  # type: ignore
        except ImportError:
            raise ImportError(
                "Vertex AI support requires google-cloud-aiplatform. "
                "Install with: pip install google-cloud-aiplatform"
            )

        model = TextEmbeddingModel.from_pretrained(self.model)
        merged_kwargs = {**self.kwargs, **kwargs}
        embeddings = model.get_embeddings([text], **merged_kwargs)
        return embeddings[0].values

    def _embed_text_genai(self, text: str, **kwargs: t.Any) -> t.List[float]:
        """Embed text using Google AI (Gemini)."""
        merged_kwargs = {**self.kwargs, **kwargs}
        result = self.client.embed_content(
            model=f"models/{self.model}", content=text, **merged_kwargs
        )
        return result["embedding"]

    async def aembed_text(self, text: str, **kwargs: t.Any) -> t.List[float]:
        """Asynchronously embed a single text using Google's embedding service.

        Google's SDK doesn't provide native async support, so we use ThreadPoolExecutor.
        """
        return await run_sync_in_async(self.embed_text, text, **kwargs)

    def embed_texts(self, texts: t.List[str], **kwargs: t.Any) -> t.List[t.List[float]]:
        """Embed multiple texts using Google's embedding service."""
        texts = validate_texts(texts)
        if not texts:
            return []

        if self.use_vertex:
            return self._embed_texts_vertex(texts, **kwargs)
        else:
            return self._embed_texts_genai(texts, **kwargs)

    def _embed_texts_vertex(
        self, texts: t.List[str], **kwargs: t.Any
    ) -> t.List[t.List[float]]:
        """Embed multiple texts using Vertex AI batch processing."""
        try:
            from vertexai.language_models import TextEmbeddingModel  # type: ignore
        except ImportError:
            raise ImportError(
                "Vertex AI support requires google-cloud-aiplatform. "
                "Install with: pip install google-cloud-aiplatform"
            )

        model = TextEmbeddingModel.from_pretrained(self.model)
        merged_kwargs = {**self.kwargs, **kwargs}
        embeddings = model.get_embeddings(texts, **merged_kwargs)
        return [emb.values for emb in embeddings]

    def _embed_texts_genai(
        self, texts: t.List[str], **kwargs: t.Any
    ) -> t.List[t.List[float]]:
        """Embed multiple texts using Google AI (Gemini).

        Google AI doesn't support batch processing, so we process individually.
        """
        return [self._embed_text_genai(text, **kwargs) for text in texts]

    async def aembed_texts(
        self, texts: t.List[str], **kwargs: t.Any
    ) -> t.List[t.List[float]]:
        """Asynchronously embed multiple texts using Google's embedding service."""
        texts = validate_texts(texts)
        if not texts:
            return []

        return await run_sync_in_async(self.embed_texts, texts, **kwargs)

    def _get_client_info(self) -> str:
        """Get client type information."""
        if self.use_vertex:
            return "<VertexAI>"
        else:
            client_type = self.client.__class__.__name__
            return f"<{client_type}>"

    def _get_key_config(self) -> str:
        """Get key configuration parameters as a string."""
        config_parts = []

        if self.use_vertex:
            config_parts.append(f"use_vertex={self.use_vertex}")
            if self.project_id:
                config_parts.append(f"project_id='{self.project_id}'")
            if self.location != "us-central1":
                config_parts.append(f"location='{self.location}'")
        else:
            config_parts.append(f"use_vertex={self.use_vertex}")

        return ", ".join(config_parts)

    def __repr__(self) -> str:
        """Return a detailed string representation of the Google embeddings."""
        client_info = self._get_client_info()
        key_config = self._get_key_config()

        base_repr = f"GoogleEmbeddings(provider='google', model='{self.model}', client={client_info}"

        if key_config:
            base_repr += f", {key_config}"

        base_repr += ")"
        return base_repr

    __str__ = __repr__

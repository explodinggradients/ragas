from __future__ import annotations

import os
import typing as t
from dataclasses import dataclass, field
from typing import List

import numpy as np
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema.embeddings import Embeddings as RagasEmbeddings

DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"


@dataclass
class HuggingfaceEmbeddings(RagasEmbeddings):
    model_name: str = DEFAULT_MODEL_NAME
    """Model name to use."""
    cache_folder: t.Optional[str] = None
    """Path to store models. 
    Can be also set by SENTENCE_TRANSFORMERS_HOME environment variable."""
    model_kwargs: t.Dict[str, t.Any] = field(default_factory=dict)
    """Keyword arguments to pass to the model."""
    encode_kwargs: t.Dict[str, t.Any] = field(default_factory=dict)

    def __post_init__(self):
        try:
            import sentence_transformers
            from transformers import AutoConfig
            from transformers.models.auto.modeling_auto import (
                MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES,
            )
        except ImportError as exc:
            raise ImportError(
                "Could not import sentence_transformers python package. "
                "Please install it with `pip install sentence-transformers`."
            ) from exc
        config = AutoConfig.from_pretrained(self.model_name)
        self.is_cross_encoder = bool(
            np.intersect1d(
                list(MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES.values()),
                config.architectures,
            )
        )

        if self.is_cross_encoder:
            self.model = sentence_transformers.CrossEncoder(
                self.model_name, **self.model_kwargs
            )
        else:
            self.model = sentence_transformers.SentenceTransformer(
                self.model_name, cache_folder=self.cache_folder, **self.model_kwargs
            )

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        from sentence_transformers.SentenceTransformer import SentenceTransformer
        from torch import Tensor

        assert isinstance(
            self.model, SentenceTransformer
        ), "Model is not of the type Bi-encoder"
        embeddings = self.model.encode(
            texts, normalize_embeddings=True, **self.encode_kwargs
        )

        assert isinstance(embeddings, Tensor)
        return embeddings.tolist()

    def predict(self, texts: List[List[str]]) -> List[List[float]]:
        from sentence_transformers.cross_encoder import CrossEncoder
        from torch import Tensor

        assert isinstance(
            self.model, CrossEncoder
        ), "Model is not of the type CrossEncoder"

        predictions = self.model.predict(texts, **self.encode_kwargs)

        assert isinstance(predictions, Tensor)
        return predictions.tolist()


def embedding_factory() -> RagasEmbeddings:
    oai_key = os.getenv("OPENAI_API_KEY", "no-key")
    openai_embeddings = OpenAIEmbeddings(openai_api_key=oai_key)
    return openai_embeddings

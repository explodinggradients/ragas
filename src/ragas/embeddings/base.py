from __future__ import annotations

import os
import typing as t
from dataclasses import field
from typing import List

import numpy as np
from langchain.embeddings import AzureOpenAIEmbeddings as BaseAzureOpenAIEmbeddings
from langchain.embeddings import OpenAIEmbeddings as BaseOpenAIEmbeddings
from langchain.schema.embeddings import Embeddings
from pydantic.dataclasses import dataclass

from ragas.exceptions import AzureOpenAIKeyNotFound, OpenAIKeyNotFound
from ragas.utils import NO_KEY

DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"


class RagasEmbeddings(Embeddings):
    def validate_api_key(self):
        """
        Validates that the api key is set for the Embeddings
        """
        pass


class OpenAIEmbeddings(BaseOpenAIEmbeddings, RagasEmbeddings):
    api_key: str = NO_KEY

    def __init__(self, api_key: str = NO_KEY):
        # api key
        key_from_env = os.getenv("OPENAI_API_KEY", NO_KEY)
        if key_from_env != NO_KEY:
            openai_api_key = key_from_env
        else:
            openai_api_key = api_key
        super(BaseOpenAIEmbeddings, self).__init__(openai_api_key=openai_api_key)
        self.api_key = openai_api_key

    def validate_api_key(self):
        if self.openai_api_key == NO_KEY:
            os_env_key = os.getenv("OPENAI_API_KEY", NO_KEY)
            if os_env_key != NO_KEY:
                self.api_key = os_env_key
            else:
                raise OpenAIKeyNotFound


class AzureOpenAIEmbeddings(BaseAzureOpenAIEmbeddings, RagasEmbeddings):
    azure_endpoint: t.Optional[str] = None
    deployment: t.Optional[str] = None
    api_version: t.Optional[str] = None
    api_key: str = NO_KEY

    def __init__(
        self,
        api_version: t.Optional[str] = None,
        azure_endpoint: t.Optional[str] = None,
        deployment: t.Optional[str] = None,
        api_key: str = NO_KEY,
    ):
        # api key
        key_from_env = os.getenv("AZURE_OPENAI_API_KEY", NO_KEY)
        if key_from_env != NO_KEY:
            openai_api_key = key_from_env
        else:
            openai_api_key = api_key

        super(BaseAzureOpenAIEmbeddings, self).__init__(
            azure_endpoint=azure_endpoint,  # type: ignore (pydantic bug I think)
            deployment=deployment,
            api_version=api_version,
            api_key=openai_api_key,
        )
        self.api_key = openai_api_key

    def validate_api_key(self):
        if self.openai_api_key == NO_KEY:
            os_env_key = os.getenv("AZURE_OPENAI_API_KEY", NO_KEY)
            if os_env_key != NO_KEY:
                self.api_key = os_env_key
            else:
                raise AzureOpenAIKeyNotFound


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

        # ensure outputs are tensors
        if "convert_to_tensor" not in self.encode_kwargs:
            self.encode_kwargs["convert_to_tensor"] = True

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
    openai_embeddings = OpenAIEmbeddings()
    return openai_embeddings

# Legacy embeddings - maintain backward compatibility
from ragas.embeddings.base import (
    BaseRagasEmbeddings,
    HuggingfaceEmbeddings,
    LangchainEmbeddingsWrapper,
    LlamaIndexEmbeddingsWrapper,
    embedding_factory,
)
from ragas.embeddings.haystack_wrapper import HaystackEmbeddingsWrapper

# Modern embeddings - new interface
from ragas.embeddings.base import BaseRagasEmbedding
from ragas.embeddings.openai_provider import OpenAIEmbeddings
from ragas.embeddings.google_provider import GoogleEmbeddings
from ragas.embeddings.litellm_provider import LiteLLMEmbeddings
from ragas.embeddings.huggingface_provider import HuggingFaceEmbeddings

# Utilities
from ragas.embeddings.utils import validate_texts, batch_texts, get_optimal_batch_size

__all__ = [
    # Legacy interface (backward compatibility)
    "BaseRagasEmbeddings",
    "HaystackEmbeddingsWrapper",
    "HuggingfaceEmbeddings",
    "LangchainEmbeddingsWrapper",
    "LlamaIndexEmbeddingsWrapper",
    "embedding_factory",
    # Modern interface
    "BaseRagasEmbedding",
    # Backward compatibility alias
    "RagasBaseEmbedding",
    "OpenAIEmbeddings",
    "GoogleEmbeddings",
    "LiteLLMEmbeddings",
    "HuggingFaceEmbeddings",
    # Utilities
    "validate_texts",
    "batch_texts",
    "get_optimal_batch_size",
]

# Backward compatibility alias
RagasBaseEmbedding = BaseRagasEmbedding

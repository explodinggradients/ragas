# Legacy embeddings - maintain backward compatibility
# Modern embeddings - new interface
from ragas.embeddings.base import (
    BaseRagasEmbedding,
    BaseRagasEmbeddings,
    HuggingfaceEmbeddings,
    LangchainEmbeddingsWrapper,
    LlamaIndexEmbeddingsWrapper,
    embedding_factory,
)
from ragas.embeddings.google_provider import GoogleEmbeddings
from ragas.embeddings.haystack_wrapper import HaystackEmbeddingsWrapper
from ragas.embeddings.huggingface_provider import HuggingFaceEmbeddings
from ragas.embeddings.litellm_provider import LiteLLMEmbeddings
from ragas.embeddings.openai_provider import OpenAIEmbeddings

# Utilities
from ragas.embeddings.utils import batch_texts, get_optimal_batch_size, validate_texts

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

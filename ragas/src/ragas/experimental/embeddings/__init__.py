from .base import BaseEmbedding, embedding_factory

# Import provider classes for direct usage
try:
    from .openai import OpenAIEmbeddings
except ImportError:
    OpenAIEmbeddings = None
try:
    from .google import GoogleEmbeddings
except ImportError:
    GoogleEmbeddings = None

try:
    from .litellm import LiteLLMEmbeddings
except ImportError:
    LiteLLMEmbeddings = None

try:
    from .huggingface import HuggingFaceEmbeddings
except ImportError:
    HuggingFaceEmbeddings = None

__all__ = [
    "BaseEmbedding",
    "OpenAIEmbeddings",
    "GoogleEmbeddings",
    "LiteLLMEmbeddings",
    "HuggingFaceEmbeddings",
    "embedding_factory",
]

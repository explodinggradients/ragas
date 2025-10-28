"""Factory functions for creating LLMs and embeddings for testing.

This module provides reusable functions for creating both legacy and modern
LLM and embedding instances. These can be used in both pytest tests (via fixtures)
and Jupyter notebooks (directly).
"""

import os
from typing import Optional


def check_api_key(provider: str = "openai") -> bool:
    """Check if required API key is set.

    Args:
        provider: The provider to check for (default: "openai")

    Returns:
        True if API key is set

    Raises:
        ValueError: If API key is not set
    """
    env_vars = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }

    env_var = env_vars.get(provider.lower())
    if not env_var:
        raise ValueError(f"Unknown provider: {provider}")

    if not os.getenv(env_var):
        raise ValueError(
            f"{env_var} environment variable not set. "
            f"Please set it before running:\n"
            f"  export {env_var}='your-api-key-here'"
        )

    return True


def create_legacy_llm(model: str = "gpt-3.5-turbo", **kwargs):
    """Create an LLM instance using the unified llm_factory.

    Args:
        model: The model name to use
        **kwargs: Additional arguments to pass to llm_factory (must include client)

    Returns:
        InstructorBaseRagasLLM instance

    Raises:
        ImportError: If llm_factory is not available
        Exception: If LLM creation fails (e.g., missing API key or client)
    """
    try:
        from ragas.llms.base import llm_factory

        if "client" not in kwargs:
            import openai

            kwargs["client"] = openai.OpenAI()

        return llm_factory(model, **kwargs)
    except ImportError as e:
        raise ImportError(f"LLM factory not available: {e}")
    except Exception as e:
        raise Exception(f"Could not create LLM (API key may be missing): {e}")


def create_modern_llm(
    provider: str = "openai",
    model: str = "gpt-3.5-turbo",
    client: Optional[any] = None,
    **kwargs,
):
    """Create an LLM instance using the unified llm_factory.

    Args:
        provider: The LLM provider (default: "openai")
        model: The model name to use
        client: Optional client instance. If None, will create AsyncOpenAI().
        **kwargs: Additional arguments to pass to llm_factory

    Returns:
        InstructorBaseRagasLLM instance

    Raises:
        ImportError: If required libraries are not available
        Exception: If LLM creation fails
    """
    try:
        from ragas.llms.base import llm_factory

        if client is None:
            if provider == "openai":
                import openai

                client = openai.AsyncOpenAI()
            else:
                raise ValueError(f"Auto-client creation not supported for {provider}")

        return llm_factory(model=model, provider=provider, client=client, **kwargs)
    except ImportError as e:
        raise ImportError(f"LLM factory not available: {e}")
    except Exception as e:
        raise Exception(f"Could not create LLM (API key may be missing): {e}")


def create_legacy_embeddings(model: str = "text-embedding-ada-002", **kwargs):
    """Create legacy embeddings for old-style metrics.

    Args:
        model: The embedding model name to use
        **kwargs: Additional arguments to pass to embedding_factory

    Returns:
        Legacy embeddings instance

    Raises:
        ImportError: If embedding_factory is not available
        Exception: If embeddings creation fails
    """
    try:
        from ragas.embeddings.base import embedding_factory

        return embedding_factory(model, **kwargs)
    except ImportError as e:
        raise ImportError(f"Embedding factory not available: {e}")
    except Exception as e:
        raise Exception(
            f"Could not create legacy embeddings (API key may be missing): {e}"
        )


def create_modern_embeddings(
    provider: str = "openai",
    model: str = "text-embedding-ada-002",
    client: Optional[any] = None,
    interface: str = "modern",
    **kwargs,
):
    """Create modern embeddings for v2 metrics.

    Args:
        provider: The embeddings provider (e.g., "openai")
        model: The embedding model name to use
        client: Optional async client instance. If None, will create one.
        interface: Interface type (default: "modern")
        **kwargs: Additional arguments to pass to embedding_factory

    Returns:
        Modern embeddings instance

    Raises:
        ImportError: If required libraries are not available
        Exception: If embeddings creation fails
    """
    try:
        from ragas.embeddings.base import embedding_factory

        # Create client if not provided
        if client is None:
            if provider == "openai":
                import openai

                client = openai.AsyncOpenAI()
            else:
                raise ValueError(f"Auto-client creation not supported for {provider}")

        return embedding_factory(
            provider=provider,
            model=model,
            client=client,
            interface=interface,
            **kwargs,
        )
    except ImportError as e:
        raise ImportError(f"OpenAI or embedding factory not available: {e}")
    except Exception as e:
        raise Exception(
            f"Could not create modern embeddings (API key may be missing): {e}"
        )


# Legacy-style factory functions for backward compatibility with langchain wrappers
def create_legacy_llm_with_langchain(model: str = "gpt-4o-mini", **kwargs):
    """Create a legacy LLM using Langchain wrapper.

    This is for compatibility with older code that uses Langchain wrappers.

    Args:
        model: The model name to use
        **kwargs: Additional arguments

    Returns:
        LangchainLLMWrapper instance
    """
    try:
        from langchain_openai import ChatOpenAI

        from ragas.llms.base import LangchainLLMWrapper

        langchain_llm = ChatOpenAI(model=model, **kwargs)
        return LangchainLLMWrapper(langchain_llm)
    except ImportError as e:
        raise ImportError(f"Langchain or LangchainLLMWrapper not available: {e}")


def create_legacy_embeddings_with_langchain(
    model: str = "text-embedding-ada-002", **kwargs
):
    """Create legacy embeddings using Langchain wrapper.

    This is for compatibility with older code that uses Langchain wrappers.

    Args:
        model: The embedding model name to use
        **kwargs: Additional arguments

    Returns:
        LangchainEmbeddingsWrapper instance
    """
    try:
        from langchain_openai import OpenAIEmbeddings

        from ragas.embeddings.base import LangchainEmbeddingsWrapper

        langchain_embeddings = OpenAIEmbeddings(model=model, **kwargs)
        return LangchainEmbeddingsWrapper(langchain_embeddings)
    except ImportError as e:
        raise ImportError(f"Langchain or LangchainEmbeddingsWrapper not available: {e}")

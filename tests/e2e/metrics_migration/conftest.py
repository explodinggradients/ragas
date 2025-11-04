"""Common fixtures for metrics migration E2E tests.

This module provides pytest fixtures that wrap the shared utility functions
from tests.utils.llm_setup for use in E2E migration tests.
"""

import pytest

from tests.utils import (
    create_legacy_embeddings,
    create_legacy_llm,
    create_modern_embeddings,
    create_modern_llm,
)


@pytest.fixture
def legacy_llm():
    """Create a test LLM for legacy metric evaluation.

    Uses legacy llm_factory for legacy implementation.
    Skips if LLM factory is not available or API key is missing.
    """
    try:
        return create_legacy_llm("gpt-3.5-turbo")
    except Exception as e:
        pytest.skip(str(e))


@pytest.fixture
def modern_llm():
    """Create a modern LLM for v2 implementation.

    Uses llm_factory with OpenAI client.
    Skips if LLM factory is not available or API key is missing.
    """
    try:
        return create_modern_llm("openai", model="gpt-3.5-turbo")
    except Exception as e:
        pytest.skip(str(e))


@pytest.fixture
def legacy_embeddings():
    """Create legacy embeddings for legacy implementation.

    Uses legacy embedding_factory interface.
    Skips if embedding factory is not available or API key is missing.
    """
    try:
        return create_legacy_embeddings("text-embedding-ada-002")
    except Exception as e:
        pytest.skip(str(e))


@pytest.fixture
def modern_embeddings():
    """Create modern embeddings for v2 implementation.

    Uses modern interface with explicit provider and client.
    Skips if OpenAI or embedding factory is not available or API key is missing.
    """
    try:
        return create_modern_embeddings(
            provider="openai",
            model="text-embedding-ada-002",
        )
    except Exception as e:
        pytest.skip(str(e))

from __future__ import annotations

import typing as t

import numpy as np
import pytest
from pydantic import BaseModel

from ragas_experimental.embeddings.base import BaseEmbedding


def pytest_configure(config):
    """
    configure pytest for experimental tests
    """
    # Extra Pytest Markers
    # add `experimental_ci`
    config.addinivalue_line(
        "markers",
        "experimental_ci: Set of tests that will be run as part of Experimental CI",
    )
    # add `e2e`
    config.addinivalue_line(
        "markers",
        "e2e: End-to-End tests for Experimental",
    )


class MockLLM:
    """Mock LLM for testing purposes"""
    
    def __init__(self):
        self.provider = "mock"
        self.model = "mock-model"
        self.is_async = True
    
    def generate(self, prompt: str, response_model: t.Type[BaseModel]) -> BaseModel:
        # Return a mock instance of the response model
        return response_model()
    
    async def agenerate(self, prompt: str, response_model: t.Type[BaseModel]) -> BaseModel:
        # Return a mock instance of the response model
        return response_model()


class MockEmbedding(BaseEmbedding):
    """Mock Embedding for testing purposes"""

    def embed_text(self, text: str, **kwargs: t.Any) -> t.List[float]:
        np.random.seed(42)  # Set seed for deterministic tests
        return np.random.rand(768).tolist()

    async def aembed_text(self, text: str, **kwargs: t.Any) -> t.List[float]:
        np.random.seed(42)  # Set seed for deterministic tests
        return np.random.rand(768).tolist()

    def embed_document(
        self, 
        text: str, 
        metadata: t.Dict[str, t.Any] = None, 
        **kwargs: t.Any
    ) -> t.List[float]:
        return self.embed_text(text, **kwargs)

    async def aembed_document(
        self, 
        text: str, 
        metadata: t.Dict[str, t.Any] = None, 
        **kwargs: t.Any
    ) -> t.List[float]:
        return await self.aembed_text(text, **kwargs)


@pytest.fixture
def mock_llm():
    return MockLLM()


@pytest.fixture
def mock_embedding():
    return MockEmbedding()
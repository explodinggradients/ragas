from __future__ import annotations

import typing as t
from pathlib import Path

import numpy as np
import pytest
from langchain_core.outputs import Generation, LLMResult
from pydantic import BaseModel

from ragas.embeddings.base import BaseRagasEmbeddings
from ragas.llms.base import BaseRagasLLM

if t.TYPE_CHECKING:
    from langchain_core.prompt_values import PromptValue

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    # Find the project root directory (where .env is located)
    project_root = Path(__file__).parent.parent
    env_file = project_root / ".env"
    if env_file.exists():
        load_dotenv(env_file)
except ImportError:
    # dotenv is optional
    pass


def pytest_configure(config):
    """
    configure pytest
    """
    # Extra Pytest Markers
    # add `ragas_ci`
    config.addinivalue_line(
        "markers",
        "ragas_ci: Set of tests that will be run as part of Ragas CI",
    )
    # add `e2e`
    config.addinivalue_line(
        "markers",
        "e2e: End-to-End tests for Ragas",
    )


class EchoLLM(BaseRagasLLM):
    def generate_text(  # type: ignore
        self,
        prompt: PromptValue,
        *args,
        **kwargs,
    ) -> LLMResult:
        return LLMResult(generations=[[Generation(text=prompt.to_string())]])

    async def agenerate_text(  # type: ignore
        self,
        prompt: PromptValue,
        *args,
        **kwargs,
    ) -> LLMResult:
        return LLMResult(generations=[[Generation(text=prompt.to_string())]])

    def is_finished(self, response: LLMResult) -> bool:
        return True


class EchoEmbedding(BaseRagasEmbeddings):
    async def aembed_documents(self, texts: t.List[str]) -> t.List[t.List[float]]:
        return [np.random.rand(768).tolist() for _ in texts]

    async def aembed_query(self, text: str) -> t.List[float]:
        return [np.random.rand(768).tolist()]

    def embed_documents(self, texts: t.List[str]) -> t.List[t.List[float]]:
        return [np.random.rand(768).tolist() for _ in texts]

    def embed_query(self, text: str) -> t.List[float]:
        return [np.random.rand(768).tolist()]


@pytest.fixture
def fake_llm():
    return EchoLLM()


@pytest.fixture
def fake_embedding():
    return EchoEmbedding()


# ====================
# Mock fixtures from experimental tests
# ====================


class MockLLM:
    """Mock LLM for testing purposes"""

    def __init__(self):
        self.provider = "mock"
        self.model = "mock-model"
        self.is_async = True

    def generate(self, prompt: str, response_model: t.Type[BaseModel]) -> BaseModel:
        # Return a mock instance of the response model
        return response_model()

    async def agenerate(
        self, prompt: str, response_model: t.Type[BaseModel]
    ) -> BaseModel:
        # Return a mock instance of the response model
        return response_model()


class MockEmbedding(BaseRagasEmbeddings):
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
        metadata: t.Optional[t.Dict[str, t.Any]] = None,
        **kwargs: t.Any,
    ) -> t.List[float]:
        return self.embed_text(text, **kwargs)

    async def aembed_document(
        self,
        text: str,
        metadata: t.Optional[t.Dict[str, t.Any]] = None,
        **kwargs: t.Any,
    ) -> t.List[float]:
        return await self.aembed_text(text, **kwargs)


@pytest.fixture
def mock_llm():
    return MockLLM()


@pytest.fixture
def mock_embedding():
    return MockEmbedding()

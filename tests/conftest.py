from __future__ import annotations

import typing as t

import numpy as np
import pytest
from langchain_core.outputs import Generation, LLMResult

from ragas.embeddings.base import BaseRagasEmbeddings
from ragas.llms.base import BaseRagasLLM

if t.TYPE_CHECKING:
    from ragas.llms.prompt import PromptValue


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

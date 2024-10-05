from __future__ import annotations

import typing as t

import pytest
from langchain_core.outputs import Generation, LLMResult

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


@pytest.fixture
def fake_llm():
    return EchoLLM()

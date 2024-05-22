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


class FakeTestLLM(BaseRagasLLM):
    def llm(self):
        return self

    def generate_text(
        self, prompt: PromptValue, n=1, temperature=1e-8, stop=None, callbacks=[]
    ):
        generations = [[Generation(text=prompt.prompt_str)] * n]
        return LLMResult(generations=generations)

    async def agenerate_text(
        self, prompt: PromptValue, n=1, temperature=1e-8, stop=None, callbacks=[]
    ):
        return self.generate_text(prompt, n, temperature, stop, callbacks)


@pytest.fixture
def fake_llm():
    return FakeTestLLM()

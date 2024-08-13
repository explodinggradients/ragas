from ragas_experimental.llms.prompt import StringPrompt
from ragas.llms.base import BaseRagasLLM
from langchain_core.outputs import LLMResult, Generation
from ragas.llms.prompt import PromptValue

import pytest


class EchoLLM(BaseRagasLLM):
    def generate_text(  # type: ignore
        self,
        prompt: PromptValue,
    ) -> LLMResult:
        return LLMResult(generations=[[Generation(text=prompt.to_string())]])

    async def agenerate_text(  # type: ignore
        self,
        prompt: PromptValue,
    ) -> LLMResult:
        return LLMResult(generations=[[Generation(text=prompt.to_string())]])


@pytest.mark.asyncio
async def test_string_prompt():
    echo_llm = EchoLLM()
    prompt = StringPrompt(llm=echo_llm)
    assert await prompt.generate("hello") == "hello"

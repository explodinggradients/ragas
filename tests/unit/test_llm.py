import os

import pytest
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema import Generation, LLMResult

from ragas.llms.base import BaseRagasLLM
from ragas.utils import NO_KEY


class TestLLM(BaseRagasLLM):
    def llm(self):
        return self

    def generate(
        self, prompts: list[ChatPromptTemplate], n=1, temperature=0, callbacks=None
    ):
        prompt_strs = [p.format() for p in prompts]
        generations = [[Generation(text=prompt_str)] * n for prompt_str in prompt_strs]
        return LLMResult(generations=generations)

    async def agenerate(
        self, prompt: ChatPromptTemplate, n=1, temperature=0, callbacks=None
    ):
        return self.generate([prompt], n, temperature, callbacks)

    def validate_api_key(self):
        if os.getenv("FAKELLM_API_KEY", NO_KEY) == NO_KEY:
            raise ValueError("FAKELLM_API_KEY not found in environment variables.")


def test_validate_api_key():
    llm = TestLLM()
    with pytest.raises(ValueError):
        llm.validate_api_key()
    os.environ["FAKELLM_API_KEY"] = "random-key-102848595"
    # just check if no error is raised
    assert llm.validate_api_key() is None

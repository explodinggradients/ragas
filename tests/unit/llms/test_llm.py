from __future__ import annotations

import typing as t

from langchain_core.outputs import Generation, LLMResult

from ragas.llms.base import BaseRagasLLM

if t.TYPE_CHECKING:
    from ragas.llms.prompt import PromptValue


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

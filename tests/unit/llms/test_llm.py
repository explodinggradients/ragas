from __future__ import annotations

import typing as t

from langchain_core.outputs import Generation, LLMResult

from ragas.llms.base import BaseRagasLLM

if t.TYPE_CHECKING:
    from langchain_core.prompt_values import PromptValue


class FakeTestLLM(BaseRagasLLM):
    def llm(self):
        return self

    def generate_text(
        self,
        prompt: PromptValue,
        n=1,
        temperature: float = 1e-8,
        stop=None,
        callbacks=[],
    ):
        generations = [[Generation(text=prompt.to_string())] * n]
        return LLMResult(generations=generations)

    async def agenerate_text(
        self,
        prompt: PromptValue,
        n=1,
        temperature: t.Optional[float] = 1e-8,
        stop=None,
        callbacks=[],
    ):
        temp_val = temperature if temperature is not None else 1e-8
        return self.generate_text(prompt, n, temp_val, stop, callbacks)

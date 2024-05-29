from __future__ import annotations

import typing as t
from ragas.llms.base import BaseRagasLLM
from ragas.run_config import RunConfig

from langchain_core.outputs import LLMResult, Generation

if t.TYPE_CHECKING:
    from ragas.llms.prompt import PromptValue

    from langchain_core.callbacks import Callbacks

    from llama_index.core.base.llms.base import BaseLLM


class LlamaIndexWrapper(BaseRagasLLM):
    """
    A Adaptor for LlamaIndex LLMs
    """

    def __init__(
        self, llama_index_llm: BaseLLM, run_config: t.Optional[RunConfig] = None
    ):
        self.li_llm = llama_index_llm
        if run_config is None:
            run_config = RunConfig()
        self.set_run_config(run_config)

    def generate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float = 1e-8,
        stop: t.Optional[t.List[str]] = None,
        callbacks: Callbacks = None,
    ) -> LLMResult:
        li_response = self.li_llm.complete(prompt.to_string())

        return LLMResult(generations=[[Generation(text=li_response.text)]])

    async def agenerate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float = 1e-8,
        stop: t.Optional[t.List[str]] = None,
        callbacks: Callbacks = None,
    ) -> LLMResult:
        return LLMResult(generations=[[Generation(text="")]])

from __future__ import annotations

import typing as t
from dataclasses import dataclass

from langchain.schema.output import Generation, LLMResult
from llama_index.llms.base import LLM as LiLLM

from ragas.async_utils import run_async_tasks
from ragas.metrics.llms.base import BaseRagasLLM, LangchainLLM

if t.TYPE_CHECKING:
    from langchain.callbacks.base import Callbacks
    from langchain.prompts import ChatPromptTemplate


@dataclass
class LlamaIndexLLM(BaseRagasLLM):
    llamaindex_llm: LiLLM

    @property
    def llm(self) -> LiLLM:
        return self.llamaindex_llm

    def generate(
        self,
        prompts: list[ChatPromptTemplate],
        n: int = 1,
        temperature: float = 0,
        callbacks: t.Optional[Callbacks] = None,
    ) -> LLMResult:
        # set temperature to 0.2 for multiple completions
        temperature = 0.2 if n > 1 else 0
        self.llm.temperature = temperature

        # get task coroutines
        tasks = []
        for p in prompts:
            tasks.extend([self.llm.acomplete(p.format()) for _ in range(n)])

        # process results to LLMResult
        # token usage is note included for now
        results = run_async_tasks(tasks)
        results2D = [results[i : i + n] for i in range(0, len(results), n)]
        generations = [
            [Generation(text=r.text) for r in result] for result in results2D
        ]
        return LLMResult(generations=generations)

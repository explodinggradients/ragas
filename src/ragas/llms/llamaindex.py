from __future__ import annotations

import typing as t

from langchain.schema.output import Generation, LLMResult

from ragas.async_utils import run_async_tasks
from ragas.llms.base import RagasLLM

if t.TYPE_CHECKING:
    from langchain.callbacks.base import Callbacks
    from langchain.prompts import ChatPromptTemplate
    try:
        from llama_index.llms.base import LLM as LiLLM
    except ImportError:
        raise ImportError(
            "llama_index must be installed to use this function. "
            "Please, install it with `pip install llama_index`."
        )


class LlamaIndexLLM(RagasLLM):
    def __init__(self, llm: LiLLM) -> None:
        self.llama_index_llm = llm

    @property
    def llm(self) -> LiLLM:
        return self.llama_index_llm

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

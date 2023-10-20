from __future__ import annotations

import typing as t
from dataclasses import dataclass

from llama_index.llms.base import LLM as LiLLM

from ragas.metrics.llms.base import BaseRagasLLM, LangchainLLM

if t.TYPE_CHECKING:
    from langchain.callbacks.manager import Callbacks
    from langchain.prompts.base import ChatPromptTemplate


@dataclass
class LlamaIndexLLM(BaseRagasLLM):
    llamaindex_llm: LiLLM

    def generate(
        self,
        prompts: list[ChatPromptTemplate],
        n: int = 1,
        temperature: float = 0,
        callbacks: t.Optional[Callbacks] = None,
    ) -> list[list[str]]:
        ...

from __future__ import annotations

import importlib.util
import typing as t

from langchain.schema.output import Generation, LLMResult

from ragas.async_utils import run_async_tasks
from ragas.llms.base import BaseRagasLLM

if t.TYPE_CHECKING:
    from langchain.callbacks.base import Callbacks
    from langchain.prompts import ChatPromptTemplate
    from openllm import LLM, HTTPClient


class OpenLLM(BaseRagasLLM):
    n_completions_supported = importlib.util.find_spec("vllm") is not None

    def __init__(self, model_id: str, **kwargs: t.Any) -> None:
        try:
            import openllm
        except ImportError:
            raise ImportError(
                "OpenLLM is not installed. Please install it using `pip install openllm`"
            )
        self._llm = openllm.LLM[t.Any, t.Any](model_id, **kwargs)

    @property
    def llm(self) -> LLM[t.Any, t.Any]:
        return self._llm

    def generate(
        self,
        prompts: list[ChatPromptTemplate],
        n: int = 1,
        temperature: float = 0,
        callbacks: t.Optional[Callbacks] = None,
    ) -> LLMResult:
        if n > 1 and not self.n_completions_supported:
            raise ValueError(
                f"Generating multiple completions is not supported by this LLM. with backend {self.llm.__llm_backend__}"
            )
        temperature = 0.2 if n > 1 else 0
        results = run_async_tasks(
            [
                self.llm.generate(p.format(), n=n, temperature=temperature)
                for p in prompts
            ]
        )
        return LLMResult(
            generations=[
                [Generation(text=r.text) for r in result]
                for result in [results[i : i + n] for i in range(0, len(results), n)]
            ]
        )


class OpenLLMAPI(BaseRagasLLM):
    n_completions_supported = True

    def __init__(self, server_url: str, **kwargs: t.Any) -> None:
        try:
            import openllm_client
        except ImportError:
            raise ImportError(
                "openllm-client is not installed. Please install it using `pip install openllm-client`"
            )
        self._llm = openllm_client.HTTPClient(server_url, **kwargs)

    @property
    def llm(self) -> HTTPClient:
        return self._llm

    def generate(
        self,
        prompts: list[ChatPromptTemplate],
        n: int = 1,
        temperature: float = 0,
        callbacks: t.Optional[Callbacks] = None,
    ) -> LLMResult:
        temperature = 0.2 if n > 1 else 0
        results = [
            self.llm.generate(p.format(), n=n, temperature=temperature) for p in prompts
        ]
        return LLMResult(
            generations=[
                [Generation(text=r.text) for r in result]
                for result in [results[i : i + n] for i in range(0, len(results), n)]
            ]
        )

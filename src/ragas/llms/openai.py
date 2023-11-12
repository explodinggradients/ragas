from __future__ import annotations

import os
import typing as t
from dataclasses import dataclass
from sys import api_version

from langchain.adapters.openai import convert_message_to_dict
from langchain.schema import Generation, LLMResult
from openai import AsyncAzureOpenAI, AsyncOpenAI

from ragas.async_utils import run_async_tasks
from ragas.llms.base import BaseRagasLLM
from ragas.llms.langchain import _compute_token_usage_langchain
from ragas.utils import NO_KEY

if t.TYPE_CHECKING:
    from langchain.callbacks.base import Callbacks
    from langchain.prompts import ChatPromptTemplate


@dataclass
class OpenAI(BaseRagasLLM):
    model: str = "gpt-3.5-turbo-16k"
    api_key: str = os.getenv("OPENAI_API_KEY", "no-key")

    @property
    def llm(self):
        return self

    def __post_init__(self):
        self.client = AsyncOpenAI(api_key=self.api_key)

    def create_llm_result(self, response) -> LLMResult:
        """Create the LLMResult from the choices and prompts."""
        if not isinstance(response, dict):
            response = response.model_dump()

        # token Usage
        token_usage = response.get("usage", {})
        llm_output = {
            "token_usage": token_usage,
            "model_name": None,
            "system_fingerprint": response.get("system_fingerprint", ""),
        }

        choices = response["choices"]
        generations = [
            Generation(
                text=choice["message"]["content"],
                generation_info=dict(
                    finish_reason=choice.get("finish_reason"),
                    logprobs=choice.get("logprobs"),
                ),
            )
            for choice in choices
        ]
        llm_output = {"token_usage": token_usage, "model_name": self.model}
        return LLMResult(generations=[generations], llm_output=llm_output)

    def generate(
        self,
        prompts: list[ChatPromptTemplate],
        n: int = 1,
        temperature: float = 0,
        callbacks: t.Optional[Callbacks] = None,
    ) -> t.Any:  # TODO: LLMResult
        llm_results = run_async_tasks(
            [self.agenerate(p, n, temperature, callbacks) for p in prompts]
        )

        generations = [r.generations[0] for r in llm_results]
        llm_output = _compute_token_usage_langchain(llm_results)
        return LLMResult(generations=generations, llm_output=llm_output)

    async def agenerate(
        self,
        prompt: ChatPromptTemplate,
        n: int = 1,
        temperature: float = 0,
        callbacks: t.Optional[Callbacks] = None,
    ) -> LLMResult:
        # TODO: use callbacks for llm generate
        completion = await self.client.chat.completions.create(
            model=self.model,
            messages=[convert_message_to_dict(m) for m in prompt.format_messages()],
            temperature=temperature,
            n=n,
        )

        return self.create_llm_result(completion)


class AzureOpenAI(OpenAI):
    api_version: str
    azure_endpoint: str

    def __init__(
        self,
        api_version: str,
        azure_endpoint: str,
        model: str,
        api_key: str = NO_KEY,
    ):
        self.api_version = api_version
        self.azure_endpoint = azure_endpoint
        super().__init__(
            model=model, api_key=os.getenv("AZURE_OPENAI_API_KEY", api_key)
        )

    def __post_init__(self):
        self.client = AsyncAzureOpenAI(
            api_version=self.api_version,
            azure_endpoint=self.azure_endpoint,
            api_key=self.api_key,
        )

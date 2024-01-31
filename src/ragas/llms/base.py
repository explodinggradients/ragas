from __future__ import annotations

import asyncio
import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import partial

from langchain_community.chat_models import ChatVertexAI
from langchain_community.llms import VertexAI
from langchain_openai.llms import AzureOpenAI, OpenAI
from langchain_openai.llms.base import BaseOpenAI
from langchain_openai.chat_models import ChatOpenAI, AzureChatOpenAI
from langchain_core.language_models import BaseLanguageModel
from langchain_core.outputs import LLMResult

from ragas.run_config import RunConfig, make_retry_wrapper, make_async_retry_wrapper

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks

    from ragas.llms.prompt import PromptValue


MULTIPLE_COMPLETION_SUPPORTED = [
    OpenAI,
    ChatOpenAI,
    AzureOpenAI,
    AzureChatOpenAI,
    ChatVertexAI,
    VertexAI,
]


def is_multiple_completion_supported(llm: BaseLanguageModel) -> bool:
    """Return whether the given LLM supports n-completion."""
    for llm_type in MULTIPLE_COMPLETION_SUPPORTED:
        if isinstance(llm, llm_type):
            return True
    return False


@dataclass
class BaseRagasLLM(ABC):
    run_config: RunConfig

    def set_run_config(self, run_config: RunConfig):
        self.run_config = run_config

    def get_temperature(self, n: int) -> float:
        """Return the temperature to use for completion based on n."""
        return 0.3 if n > 1 else 1e-8

    @abstractmethod
    def generate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float = 1e-8,
        stop: t.Optional[t.List[str]] = None,
        callbacks: Callbacks = [],
    ) -> LLMResult:
        ...

    @abstractmethod
    async def agenerate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float = 1e-8,
        stop: t.Optional[t.List[str]] = None,
        callbacks: Callbacks = [],
    ) -> LLMResult:
        ...

    async def generate(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float = 1e-8,
        stop: t.Optional[t.List[str]] = None,
        callbacks: Callbacks = [],
        is_async: bool = True,
    ) -> LLMResult:
        """Generate text using the given event loop."""
        if is_async:
            with_retry = make_async_retry_wrapper(self.run_config)
            return await with_retry(
                self.agenerate_text,
                prompt=prompt,
                n=n,
                temperature=temperature,
                stop=stop,
                callbacks=callbacks,
            )
        else:
            loop = asyncio.get_event_loop()
            with_retry = make_retry_wrapper(self.run_config)
            generate_text = partial(
                with_retry(self.generate_text),
                prompt=prompt,
                n=n,
                temperature=temperature,
                stop=stop,
                callbacks=callbacks,
            )
            return await loop.run_in_executor(None, generate_text)


class LangchainLLMWrapper(BaseRagasLLM):
    """
    A simple base class for RagasLLMs that is based on Langchain's BaseLanguageModel
    interface. it implements 2 functions:
    - generate_text: for generating text from a given PromptValue
    - agenerate_text: for generating text from a given PromptValue asynchronously
    """

    def __init__(
        self, langchain_llm: BaseLanguageModel, run_config: t.Optional[RunConfig]
    ):
        self.langchain_llm = langchain_llm
        if run_config is None:
            run_config = RunConfig()
        self.run_config = run_config

    def generate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float = 1e-8,
        stop: t.Optional[t.List[str]] = None,
        callbacks: t.Optional[Callbacks] = None,
    ) -> LLMResult:
        temperature = self.get_temperature(n=n)
        if is_multiple_completion_supported(self.langchain_llm):
            return self.langchain_llm.generate_prompt(
                prompts=[prompt],
                n=n,
                temperature=temperature,
                stop=stop,
                callbacks=callbacks,
            )
        else:
            result = self.langchain_llm.generate_prompt(
                prompts=[prompt] * n,
                temperature=temperature,
                stop=stop,
                callbacks=callbacks,
            )
            # make LLMResult.generation appear as if it was n_completions
            # note that LLMResult.runs is still a list that represents each run
            generations = [[g[0] for g in result.generations]]
            result.generations = generations
            return result

    async def agenerate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float = 1e-8,
        stop: t.Optional[t.List[str]] = None,
        callbacks: t.Optional[Callbacks] = None,
    ) -> LLMResult:
        temperature = self.get_temperature(n=n)
        if is_multiple_completion_supported(self.langchain_llm):
            return await self.langchain_llm.agenerate_prompt(
                prompts=[prompt],
                n=n,
                temperature=temperature,
                stop=stop,
                callbacks=callbacks,
            )
        else:
            result = await self.langchain_llm.agenerate_prompt(
                prompts=[prompt] * n,
                temperature=temperature,
                stop=stop,
                callbacks=callbacks,
            )
            # make LLMResult.generation appear as if it was n_completions
            # note that LLMResult.runs is still a list that represents each run
            generations = [[g[0] for g in result.generations]]
            result.generations = generations
            return result

    def set_run_config(self, run_config: RunConfig):
        self.run_config = run_config
        # configure timeout for the underlying LLM in case of OpenAI
        if isinstance(self.langchain_llm, BaseOpenAI) or isinstance(
            self.langchain_llm, ChatOpenAI
        ):
            self.langchain_llm.request_timeout = run_config.timeout


def llm_factory(
    model: str = "gpt-3.5-turbo-16k", run_config: t.Optional[RunConfig] = None
) -> BaseRagasLLM:
    timeout = None
    if run_config is not None:
        timeout = run_config.timeout
    openai_model = ChatOpenAI(model=model, request_timeout=timeout)  # type: ignore
    return LangchainLLMWrapper(openai_model, run_config)

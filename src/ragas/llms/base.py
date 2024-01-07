from __future__ import annotations

import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass

from langchain.chat_models import AzureChatOpenAI, ChatOpenAI, ChatVertexAI
from langchain.llms import AzureOpenAI, OpenAI, VertexAI
from langchain_core.language_models import BaseLanguageModel
from langchain_core.outputs import LLMResult

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks
    from langchain_core.prompts import ChatPromptTemplate

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

    # TODO: remove after testset generator is refactored
    def generate_text_with_hmpt(
        self,
        prompts: t.List[ChatPromptTemplate],
        n: int = 1,
        temperature: float = 1e-8,
        stop: t.Optional[t.List[str]] = None,
        callbacks: Callbacks = [],
    ) -> LLMResult:
        from ragas.llms.prompt import PromptValue

        prompt = PromptValue(prompt_str=prompts[0].format())
        return self.generate_text(prompt, n, temperature, stop, callbacks)


@dataclass
class LangchainLLMWrapper(BaseRagasLLM):
    """
    A simple base class for RagasLLMs that is based on Langchain's BaseLanguageModel
    interface. it implements 2 functions:
    - generate_text: for generating text from a given PromptValue
    - agenerate_text: for generating text from a given PromptValue asynchronously
    """

    langchain_llm: BaseLanguageModel

    def generate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float = 1e-8,
        stop: t.Optional[t.List[str]] = None,
        callbacks: t.Optional[Callbacks] = None,
    ) -> LLMResult:
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

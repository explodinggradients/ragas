from __future__ import annotations

import typing as t
from abc import ABC, abstractmethod

from langchain.chat_models import AzureChatOpenAI, ChatOpenAI
from langchain.chat_models.base import BaseChatModel
from langchain.llms import AzureOpenAI, OpenAI
from langchain.llms.base import BaseLLM
from langchain.prompts import ChatPromptTemplate
from langchain.schema import LLMResult

from ragas.async_utils import run_async_tasks

if t.TYPE_CHECKING:
    from langchain.callbacks.base import Callbacks


def isOpenAI(llm: BaseLLM | BaseChatModel) -> bool:
    return isinstance(llm, OpenAI) or isinstance(llm, ChatOpenAI)


# have to specify it twice for runtime and static checks
MULTIPLE_COMPLETION_SUPPORTED = [OpenAI, ChatOpenAI, AzureOpenAI, AzureChatOpenAI]
MultipleCompletionSupportedLLM = t.Union[
    OpenAI, ChatOpenAI, AzureOpenAI, AzureChatOpenAI
]


def multiple_completion_supported(llm: BaseLLM | BaseChatModel) -> bool:
    for model in MULTIPLE_COMPLETION_SUPPORTED:
        if isinstance(llm, model):
            return True
    return False


def generate(
    prompts: list[ChatPromptTemplate],
    llm: BaseLLM | BaseChatModel,
    n: int = 1,
    temperature: float = 0,
    callbacks: t.Optional[Callbacks] = None,
) -> LLMResult:
    old_n: int = 1
    n_swapped = False
    llm.temperature = temperature
    if n is not None:
        if multiple_completion_supported(llm):
            llm = t.cast(MultipleCompletionSupportedLLM, llm)
            old_n = llm.n  # type: ignore (n is not found in valid)
            llm.n = n
            n_swapped = True
        else:
            raise Exception(
                f"n={n} was passed to generate but the LLM {llm} does not support it."
                " Raise an issue if you want support for {llm}."
            )
    if isinstance(llm, BaseLLM):
        ps = [p.format() for p in prompts]
        result = llm.generate(ps, callbacks=callbacks)
    elif isinstance(llm, BaseChatModel):
        ps = [p.format_messages() for p in prompts]
        result = llm.generate(ps, callbacks=callbacks)

    if multiple_completion_supported(llm) and n_swapped:
        llm = t.cast(MultipleCompletionSupportedLLM, llm)
        llm.n = old_n

    return result


class BaseRagasLLM(ABC):
    """
    BaseLLM is the base class for all LLMs. It provides a consistent interface for other
    classes that interact with LLMs like Langchains, LlamaIndex, LiteLLM etc. Handles
    multiple_completions even if not supported by the LLM.

    It currently takes in ChatPromptTemplates and returns LLMResults which are Langchain
    primitives.
    """

    # supports multiple compeletions for the given prompt
    n_completions_supported: bool = False

    @abstractmethod
    def generate(
        self,
        prompts: list[str],
        n: int = 1,
        temperature: float = 0,
        callbacks: t.Optional[Callbacks] = None,
    ) -> list[list[str]]:
        ...


class LangchainLLM(BaseRagasLLM):
    n_completions_supported: bool = True

    def __init__(self, llm: BaseLLM | BaseChatModel):
        self.llm = llm

    @staticmethod
    def llm_supports_completions(llm):
        completion_supported_llms = [OpenAI, ChatOpenAI, AzureOpenAI, AzureChatOpenAI]
        for llm_type in completion_supported_llms:
            if isinstance(llm, llm_type):
                return True

    def generate_multiple_completions(
        self,
        prompts: list[ChatPromptTemplate],
        n: t.Optional[int] = None,
        callbacks: t.Optional[Callbacks] = None,
    ) -> LLMResult:
        old_n = self.llm.n  # type: ignore (this function will only run for models with n)
        self.llm.n = n
        if isinstance(self.llm, BaseLLM):
            ps = [p.format() for p in prompts]
            result = self.llm.generate(ps, callbacks=callbacks)
        else:  # if BaseChatModel
            ps = [p.format_messages() for p in prompts]
            result = self.llm.generate(ps, callbacks=callbacks)
        self.llm.n = old_n

        return result

    async def generate_completions(
        self,
        prompts: list[ChatPromptTemplate],
        callbacks: t.Optional[Callbacks] = None,
    ) -> LLMResult:
        if isinstance(self.llm, BaseLLM):
            ps = [p.format() for p in prompts]
            result = await self.llm.agenerate(ps, callbacks=callbacks)
        else:  # if BaseChatModel
            ps = [p.format_messages() for p in prompts]
            result = await self.llm.agenerate(ps, callbacks=callbacks)

        return result

    def generate(
        self,
        prompts: list[ChatPromptTemplate],
        n: int = 1,
        temperature: float = 0,
        callbacks: t.Optional[Callbacks] = None,
    ) -> LLMResult:
        self.llm.temperature = temperature

        if self.llm_supports_completions(self.llm):
            return self.generate_multiple_completions(prompts, n, callbacks)
        else:  # call generate_completions n times to mimic multiple completions
            list_llmresults = run_async_tasks(
                [self.generate_completions(prompts, callbacks) for _ in range(n)]
            )

            # fill results as if the LLM supported multiple completions
            generations = []
            for i in range(len(prompts)):
                completions = []
                for result in list_llmresults:
                    completions.append(result.generations[i][0])
                generations.append(completions)

            # compute total token usage by adding individual token usage
            llm_output = list_llmresults[0].llm_output
            if "token_usage" in llm_output:
                sum_prompt_tokens = 0
                sum_completion_tokens = 0
                sum_total_tokens = 0
                for result in list_llmresults:
                    token_usage = result.llm_output["token_usage"]
                    sum_prompt_tokens += token_usage["prompt_tokens"]
                    sum_completion_tokens += token_usage["completion_tokens"]
                    sum_total_tokens += token_usage["total_tokens"]

                llm_output["token_usage"] = {
                    "prompt_tokens": sum_prompt_tokens,
                    "completion_tokens": sum_completion_tokens,
                    "sum_total_tokens": sum_total_tokens,
                }

            return LLMResult(generations=generations, llm_output=llm_output)

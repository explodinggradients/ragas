from __future__ import annotations

import typing as t

from langchain.chat_models import AzureChatOpenAI, ChatOpenAI
from langchain.chat_models.base import BaseChatModel
from langchain.llms import AzureOpenAI, OpenAI
from langchain.llms.base import BaseLLM
from langchain.prompts import ChatPromptTemplate
from langchain.schema import LLMResult

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

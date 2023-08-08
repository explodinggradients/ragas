from __future__ import annotations

import typing as t

from langchain.chat_models import ChatOpenAI
from langchain.chat_models.base import BaseChatModel
from langchain.llms import OpenAI
from langchain.llms.base import BaseLLM
from langchain.prompts import ChatPromptTemplate
from langchain.schema import LLMResult

if t.TYPE_CHECKING:
    from langchain.callbacks.base import Callbacks


def isOpenAI(llm: BaseLLM | BaseChatModel) -> bool:
    return isinstance(llm, OpenAI) or isinstance(llm, ChatOpenAI)


def generate(
    prompts: list[ChatPromptTemplate],
    llm: BaseLLM | BaseChatModel,
    n: t.Optional[int] = None,
    temperature: float = 0,
    callbacks: t.Optional[Callbacks] = None,
) -> LLMResult:
    old_n = None
    n_swapped = False
    llm.temperature = temperature
    if n is not None:
        if isinstance(llm, OpenAI) or isinstance(llm, ChatOpenAI):
            old_n = llm.n
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

    if (isinstance(llm, OpenAI) or isinstance(llm, ChatOpenAI)) and n_swapped:
        llm.n = old_n  # type: ignore

    return result

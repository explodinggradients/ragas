from __future__ import annotations

import logging
import os

import openai
from langchain.chat_models.base import BaseChatModel
from langchain.llms.base import BaseLLM
from langchain.prompts import ChatPromptTemplate
from langchain.schema import LLMResult


def generate(
    prompts: list[ChatPromptTemplate], llm: BaseLLM | BaseChatModel
) -> LLMResult:
    if isinstance(llm, BaseLLM):
        ps = [p.format() for p in prompts]
        return llm.generate(ps)
    elif isinstance(llm, BaseChatModel):
        ps = [p.format_messages() for p in prompts]
        return llm.generate(ps)


# each of these calls have to check for
# https://platform.openai.com/docs/guides/error-codes/api-errors
# and handle it gracefully
def openai_completion(prompts: list[str], **kwargs):
    """
    TODOs

    - what happens when backoff fails?
    """
    response = openai.Completion.create(
        model=kwargs.get("model", "text-davinci-003"),
        prompt=prompts,
        temperature=kwargs.get("temperature", 0),
        top_p=kwargs.get("top_p", 1),
        frequency_penalty=kwargs.get("frequency_penalty", 0.0),
        presence_penalty=kwargs.get("presence_penalty", 0.0),
        max_tokens=kwargs.get("max_tokens", 500),
        logprobs=kwargs.get("logprobs", 1),
        n=kwargs.get("n", 1),
    )

    return response


# TODO: make this work
def openai_completion_async(prompts: list[str], **kwargs):
    response = openai.Completion.acreate(
        model=kwargs.get("model", "text-davinci-003"),
        prompt=prompts,
        temperature=kwargs.get("temperature", 0),
        top_p=kwargs.get("top_p", 1),
        frequency_penalty=kwargs.get("frequency_penalty", 0.0),
        presence_penalty=kwargs.get("presence_penalty", 0.0),
        max_tokens=kwargs.get("max_tokens", 500),
        logprobs=kwargs.get("logprobs", 1),
        n=kwargs.get("n", 1),
    )
    return response

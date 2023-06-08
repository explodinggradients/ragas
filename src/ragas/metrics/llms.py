from __future__ import annotations

import logging
import os

import backoff
import openai
from openai.error import RateLimitError

openai.api_key = os.environ.get("OPENAI_API_KEY")

# TODO better way of logging backoffs
logging.getLogger("backoff").addHandler(logging.StreamHandler())


# each of these calls have to check for
# https://platform.openai.com/docs/guides/error-codes/api-errors
# and handle it gracefully
@backoff.on_exception(backoff.expo, RateLimitError, max_tries=5)
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

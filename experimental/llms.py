import os

import openai

openai.api_key = os.environ.get("OPENAI_API_KEY")


# each of these calls have to check for
# https://platform.openai.com/docs/guides/error-codes/api-errors
# and handle it gracefully
def llm(prompts: list[str], **kwargs):
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


def llm_async(prompt, **kwargs):
    response = openai.Completion.acreate(
        model=kwargs.get("model", "text-davinci-003"),
        prompt=prompt,
        temperature=kwargs.get("temperature", 0),
        top_p=kwargs.get("top_p", 1),
        frequency_penalty=kwargs.get("frequency_penalty", 0.0),
        presence_penalty=kwargs.get("presence_penalty", 0.0),
        max_tokens=kwargs.get("max_tokens", 500),
        logprobs=kwargs.get("logprobs", 1),
        n=kwargs.get("n", 1),
    )
    return response

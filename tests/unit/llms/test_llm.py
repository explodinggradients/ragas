from __future__ import annotations

import os

import pytest
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema import Generation, LLMResult

from ragas.embeddings import AzureOpenAIEmbeddings, OpenAIEmbeddings
from ragas.llms.base import RagasLLM
from ragas.llms.openai import (
    AzureOpenAI,
    AzureOpenAIKeyNotFound,
    OpenAI,
    OpenAIKeyNotFound,
)
from ragas.utils import NO_KEY


class TestLLM(RagasLLM):
    def llm(self):
        return self

    def generate(
        self, prompts: list[ChatPromptTemplate], n=1, temperature=0, callbacks=None
    ):
        prompt_strs = [p.format() for p in prompts]
        generations = [[Generation(text=prompt_str)] * n for prompt_str in prompt_strs]
        return LLMResult(generations=generations)

    async def agenerate(
        self, prompt: ChatPromptTemplate, n=1, temperature=0, callbacks=None
    ):
        return self.generate([prompt], n, temperature, callbacks)

    def validate_api_key(self):
        if os.getenv("FAKELLM_API_KEY", NO_KEY) == NO_KEY:
            raise ValueError("FAKELLM_API_KEY not found in environment variables.")


def test_validate_api_key():
    llm = TestLLM()
    with pytest.raises(ValueError):
        llm.validate_api_key()
    os.environ["FAKELLM_API_KEY"] = "random-key-102848595"
    # just check if no error is raised
    assert llm.validate_api_key() is None


def openai_llm_factory(with_api_key):
    if with_api_key:
        api_key = "random-key-102848595"
        return OpenAI(api_key=api_key), api_key
    else:
        return OpenAI()


def openai_embedding_factory(with_api_key):
    if with_api_key:
        api_key = "random-key-102848595"
        return OpenAIEmbeddings(api_key=api_key), api_key
    else:
        return OpenAIEmbeddings()


def azure_llm_factory(with_api_key):
    if with_api_key:
        api_key = "random-key-102848595"
        return (
            AzureOpenAI(
                api_version="2020-09-03",
                api_key=api_key,
                azure_endpoint="https://api.labs.cognitive.microsofttranslator.com",
                deployment="en-fr",
            ),
            api_key,
        )
    else:
        return AzureOpenAI(
            azure_endpoint="https://api.labs.cognitive.microsofttranslator.com",
            deployment="en-fr",
            api_version="2020-09-03",
        )


def azure_embed_factory(with_api_key):
    if with_api_key:
        api_key = "random-key-102848595"
        return (
            AzureOpenAIEmbeddings(
                api_version="2020-09-03",
                api_key=api_key,
                azure_endpoint="https://api.labs.cognitive.microsofttranslator.com",
                deployment="en-fr",
            ),
            api_key,
        )
    else:
        return AzureOpenAIEmbeddings(
            azure_endpoint="https://api.labs.cognitive.microsofttranslator.com",
            deployment="en-fr",
            api_version="2020-09-03",
        )


@pytest.mark.parametrize(
    "factory, key_not_found_exception, environ_key",
    [
        (openai_llm_factory, OpenAIKeyNotFound, "OPENAI_API_KEY"),
        (azure_llm_factory, AzureOpenAIKeyNotFound, "AZURE_OPENAI_API_KEY"),
        (openai_embedding_factory, OpenAIKeyNotFound, "OPENAI_API_KEY"),
        (azure_embed_factory, AzureOpenAIKeyNotFound, "AZURE_OPENAI_API_KEY"),
    ],
)
def test_validate_api_key_for_different_llms(
    factory, key_not_found_exception, environ_key
):
    # load key from environment variables
    if environ_key in os.environ:
        os.environ.pop(environ_key)
    obj = factory(with_api_key=False)
    with pytest.raises(key_not_found_exception):
        obj.validate_api_key()
    os.environ[environ_key] = "random-key-102848595"
    obj = factory(with_api_key=False)
    assert obj.validate_api_key() is None

    # load key which is passed as argument
    if environ_key in os.environ:
        os.environ.pop(environ_key)
    obj, _ = factory(with_api_key=True)
    assert obj.validate_api_key() is None

    # assert order of precedence
    os.environ[environ_key] = "random-key-102848595"
    obj, api_key = factory(with_api_key=True)
    assert obj.validate_api_key
    assert obj.api_key == api_key

    # assert loading key from environment variables after instantiation
    if environ_key in os.environ:
        os.environ.pop(environ_key)
    obj = factory(with_api_key=False)
    os.environ[environ_key] = "random-key-102848595"
    assert obj.validate_api_key() is None
    assert obj.api_key == "random-key-102848595"

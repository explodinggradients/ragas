from ragas_experimental.llms.prompt import StringPrompt
from ragas.llms.base import BaseRagasLLM
from langchain_core.outputs import LLMResult, Generation
from ragas.llms.prompt import PromptValue

import pytest


class EchoLLM(BaseRagasLLM):
    def generate_text(  # type: ignore
        self,
        prompt: PromptValue,
    ) -> LLMResult:
        return LLMResult(generations=[[Generation(text=prompt.to_string())]])

    async def agenerate_text(  # type: ignore
        self,
        prompt: PromptValue,
    ) -> LLMResult:
        return LLMResult(generations=[[Generation(text=prompt.to_string())]])


@pytest.mark.asyncio
async def test_string_prompt():
    echo_llm = EchoLLM()
    prompt = StringPrompt(llm=echo_llm)
    assert await prompt.generate("hello") == "hello"


expected_generate_output_signature = """\
Please return the output in the following JSON format based on the StringIO model:
{
    "text": "str"
}\
"""


def test_process_fields():
    from ragas_experimental.llms.prompt import PydanticPrompt, StringIO
    from pydantic import BaseModel
    from enum import Enum

    class Categories(str, Enum):
        science = "science"
        commerce = "commerce"
        agriculture = "agriculture"
        economics = "economics"

    class InputModel(BaseModel):
        category: Categories

    class JokeGenerator(PydanticPrompt[InputModel, StringIO]):
        instruction = "Generate a joke in the category of {category}."

    echo_llm = EchoLLM()
    p = JokeGenerator(llm=echo_llm)
    generation = p.generate_output_signature(StringIO)

    assert expected_generate_output_signature == generation


@pytest.mark.asyncio
async def test_pydantic_prompt_io():
    from ragas_experimental.llms.prompt import (
        PydanticPrompt,
        StringIO,
    )

    class Prompt(PydanticPrompt[StringIO, StringIO]):
        instruction = ""
        input_model = StringIO
        output_model = StringIO

    llm = EchoLLM()
    p = Prompt(llm=llm)
    assert p.input_model == StringIO
    assert p.output_model == StringIO

    assert p.generate_examples() == ""


def test_pydantic_prompt_examples():
    from ragas_experimental.llms.prompt import (
        PydanticPrompt,
        StringIO,
    )

    class Prompt(PydanticPrompt[StringIO, StringIO]):
        instruction = ""
        input_model = StringIO
        output_model = StringIO
        examples = [
            (StringIO(text="hello"), StringIO(text="hello")),
            (StringIO(text="world"), StringIO(text="world")),
        ]

    llm = EchoLLM()
    p = Prompt(llm=llm)
    assert p.generate_examples() == "hello -> hello\nworld -> world"

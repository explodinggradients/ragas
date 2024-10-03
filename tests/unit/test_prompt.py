import pytest
from langchain_core.outputs import Generation, LLMResult

from ragas.llms.base import BaseRagasLLM
from ragas.llms.prompt import PromptValue
from ragas.prompt import StringIO, StringPrompt
from ragas.run_config import RunConfig


class EchoLLM(BaseRagasLLM):
    def generate_text(  # type: ignore
        self,
        prompt: PromptValue,
        *args,
        **kwargs,
    ) -> LLMResult:
        return LLMResult(generations=[[Generation(text=prompt.to_string())]])

    async def agenerate_text(  # type: ignore
        self,
        prompt: PromptValue,
        *args,
        **kwargs,
    ) -> LLMResult:
        return LLMResult(generations=[[Generation(text=prompt.to_string())]])


@pytest.mark.asyncio
async def test_string_prompt():
    echo_llm = EchoLLM(run_config=RunConfig())
    prompt = StringPrompt()
    assert await prompt.generate(data="hello", llm=echo_llm) == "hello"
    assert prompt.name == "string_prompt"


expected_generate_output_signature = """\
Please return the output in the following JSON format based on the StringIO model:
{
    "text": "str"
}\
"""


def test_process_fields():
    from enum import Enum

    from pydantic import BaseModel

    from ragas.prompt import PydanticPrompt, StringIO

    class Categories(str, Enum):
        science = "science"
        commerce = "commerce"
        agriculture = "agriculture"
        economics = "economics"

    class InputModel(BaseModel):
        category: Categories

    class JokeGenerator(PydanticPrompt[InputModel, StringIO]):
        instruction = "Generate a joke in the category of {category}."
        output_model = StringIO

    p = JokeGenerator()
    _ = p._generate_output_signature()

    # assert expected_generate_output_signature == generation


@pytest.mark.asyncio
async def test_pydantic_prompt_io():
    from ragas.prompt import PydanticPrompt, StringIO

    class Prompt(PydanticPrompt[StringIO, StringIO]):
        instruction = ""
        input_model = StringIO
        output_model = StringIO

    p = Prompt()
    assert p.input_model == StringIO
    assert p.output_model == StringIO

    assert p._generate_examples() == ""


def test_pydantic_prompt_examples():
    from ragas.prompt import PydanticPrompt

    class Prompt(PydanticPrompt[StringIO, StringIO]):
        instruction = ""
        input_model = StringIO
        output_model = StringIO
        examples = [
            (StringIO(text="hello"), StringIO(text="hello")),
            (StringIO(text="world"), StringIO(text="world")),
        ]

    _ = Prompt()
    # assert p.generate_examples() == "hello -> hello\nworld -> world"


def test_prompt_hash():
    from ragas.prompt import StringPrompt

    class Prompt(StringPrompt):
        instruction = "You are a helpful assistant."

    p = Prompt()
    assert hash(p) == hash(p)
    p.instruction = "You are a helpful assistant. And some more"
    # assert hash(p) != hash(p)

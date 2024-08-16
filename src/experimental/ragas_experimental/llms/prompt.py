from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
import typing as t

from ragas.llms.output_parser import RagasoutputParser
from ragas.llms.prompt import PromptValue

# Check Pydantic version
from pydantic import BaseModel
import pydantic

if t.TYPE_CHECKING:
    from ragas.llms.base import BaseRagasLLM

PYDANTIC_V2 = pydantic.VERSION.startswith("2.")


class BasePrompt(ABC):
    def __init__(self, llm):
        self.llm: BaseRagasLLM = llm

    @abstractmethod
    async def generate(self, data: t.Any) -> t.Any:
        pass


def model_to_dict(
    model: BaseModel,
    by_alias: bool = False,
    exclude_unset: bool = False,
    exclude_defaults: bool = False,
) -> t.Dict[str, t.Any]:
    if PYDANTIC_V2:
        return model.model_dump(  # type: ignore
            by_alias=by_alias,
            exclude_unset=exclude_unset,
            exclude_defaults=exclude_defaults,
        )
    else:
        return model.dict(
            by_alias=by_alias,
            exclude_unset=exclude_unset,
            exclude_defaults=exclude_defaults,
        )


def to_json(model: t.Any, indent: int = 4) -> str:
    if PYDANTIC_V2:
        # Pydantic 2.x
        return model.model_dump_json(indent=indent)
    else:
        # Pydantic 1.x
        return model.json(indent=indent)


def model_to_json_schema(model: t.Type[BaseModel]) -> dict:
    if PYDANTIC_V2:
        return model.model_json_schema()
    else:
        return model.schema_json()

InputModel = t.TypeVar("InputModel", bound=BaseModel)
OutputModel = t.TypeVar("OutputModel", bound=BaseModel)


class StringIO(BaseModel):
    text: str


class PydanticPrompt(BasePrompt, t.Generic[InputModel, OutputModel]):
    input_model: t.Type[InputModel]
    output_model: t.Type[OutputModel]
    instruction: str
    examples: t.List[t.Tuple[InputModel, OutputModel]] = []

    def generate_output_signature(self, indent: int = 4) -> str:
        schema = model_to_json_schema(self.output_model)
        return (
            f"Please return the output in a JSON format that complies with the following schema as specified in JSON Schema and OpenAPI:\n"
            f"{schema}"
        )

    async def from_llm(self, prompt_value: PromptValue) -> OutputModel:
        resp = await self.llm.generate(prompt_value)
        resp_text = resp.generations[0][0].text
        print(resp_text)
        parser = RagasoutputParser(pydantic_object=self.output_model)
        answer = await parser.aparse(resp_text, prompt_value, self.llm, max_retries=3)

        # TODO: make sure RagasOutputPraser returns the same type as OutputModel
        return answer  # type: ignore

    def generate_examples(self):
        if self.examples:
            example_strings = []
            for e in self.examples:
                input_data, output_data = e
                example_strings.append(
                    self.instruction.format(**model_to_dict(input_data))
                    + "\n"
                    + to_json(output_data, indent=4)
                )

            return (
                "These are some examples to show how to perform the above instruction\n"
                + "\n\n".join(example_strings)
            )
        # if no examples are provided
        else:
            return ""

    def to_string(self, data: InputModel) -> str:
        # this needs a check
        instruction_str = self.instruction.format(**model_to_dict(data))
        json_format = self.generate_output_signature(self.output_model)
        examples_str = self.generate_examples()
        return instruction_str + "\n" + examples_str + "\n" + json_format

    async def generate(self, data: InputModel) -> OutputModel:
        prompt_value = PromptValue(prompt_str=self.to_string(data))
        result: OutputModel = await self.from_llm(prompt_value)
        return result


class StringPrompt(BasePrompt):
    async def generate(self, data: str) -> str:
        prompt_value = PromptValue(prompt_str=data)
        llm_result = await self.llm.agenerate_text(prompt_value)
        return llm_result.generations[0][0].text
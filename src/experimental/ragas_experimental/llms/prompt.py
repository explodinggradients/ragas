from abc import ABC, abstractmethod
import typing as t
import json
from ragas.llms.prompt import PromptValue
from ragas.llms.output_parser import RagasoutputParser

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


def process_field(field_type: t.Any, level: int) -> str:
    if hasattr(field_type, "__origin__"):
        if field_type.__origin__ is list:
            return f"List[{process_field(field_type.__args__[0], level)}]"
        elif field_type.__origin__ is t.Optional:
            return f"Optional[{process_field(field_type.__args__[0], level)}]"
    elif hasattr(field_type, "__fields__"):
        return (
            "{\n" + process_fields(field_type.__fields__, level + 1) + " " * level + "}"
        )
    else:
        return f"{field_type.__name__}"


def process_fields(fields: t.Dict[str, t.Any], level: int) -> str:
    field_str = ""
    for field_name, field in fields.items():
        field_str += (
            " " * level + f'"{field_name}": "{process_field(field.type_, level)}",\n'
        )
    return field_str.rstrip(",\n") + "\n"


InputModel = t.TypeVar("InputModel", bound=BaseModel)
OutputModel = t.TypeVar("OutputModel", bound=BaseModel)


class PydanticPrompt(BasePrompt, t.Generic[InputModel, OutputModel]):
    input_model: type[InputModel]
    output_model: type[OutputModel]
    instruction: str
    examples: t.List[t.Tuple[t.Any, t.Any]] = []

    def generate_output_signature(self, model: BaseModel, indent: int = 4) -> str:
        model_name = model.__class__.__name__
        fields = model.__fields__

        instruction = f"Please return the output in the following JSON format based on the {model_name} model:\n{{\n"
        instruction += process_fields(fields, indent)
        instruction += "}"

        return instruction

    async def from_llm(self, prompt_value: PromptValue) -> OutputModel:
        # TODO: implement call to LLM and output parsing
        # for openai, anthropic and ollama
        resp = self.llm.invoke(prompt_value)

        parser = RagasoutputParser(pydantic_object=self.output_model)
        answer = await parser.aparse(
            resp.content, prompt_value, self.llm, max_retries=3
        )

        # TODO: make sure RagasOutputPraser returns the same type as OutputModel
        return answer  # type: ignore

    def generate_examples(self):
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

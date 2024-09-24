from __future__ import annotations

import typing as t
from abc import ABC, abstractmethod

import pydantic

# Check Pydantic version
from pydantic import BaseModel

from ragas.llms.output_parser import RagasoutputParser
from ragas.llms.prompt import PromptValue

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks

    from ragas.llms.base import BaseRagasLLM

PYDANTIC_V2 = pydantic.VERSION.startswith("2.")


class BasePrompt(ABC):
    @abstractmethod
    async def generate(
        self,
        llm: BaseRagasLLM,
        data: t.Any,
        n: int = 1,
        temperature: t.Optional[float] = None,
        stop: t.Optional[t.List[str]] = None,
        callbacks: Callbacks = [],
    ) -> t.Any:
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


def model_to_json_schema(model: t.Type[BaseModel]) -> str:
    if PYDANTIC_V2:
        # NOTE: this is not the same as model.schema_json()
        return model.model_json_schema()  # type: ignore
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

    def generate_instruction(self) -> str:
        return self.instruction

    def generate_output_signature(self, indent: int = 4) -> str:
        schema = model_to_json_schema(self.output_model)
        return (
            f"Please return the output in a JSON format that complies with the "
            f"following schema as specified in JSON Schema and OpenAPI specification:\n"
            f"{schema}"
        )

    def generate_examples(self):
        if self.examples:
            example_strings = []
            for e in self.examples:
                input_data, output_data = e
                example_strings.append(
                    self.instruction
                    + "\n"
                    + "input: "
                    + to_json(input_data, indent=4)
                    + "\n"
                    + "output: "
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
        return (
            self.generate_instruction()
            + "\n"
            + self.generate_output_signature()
            + "\n"
            + self.generate_examples()
            + "\nNow perform the above instruction with the following input\n"
            + "input: "
            + to_json(data, indent=4)
            + "\n"
            + "output: "
        )

    async def generate(
        self,
        llm: BaseRagasLLM,
        data: InputModel,
        n: int = 1,
        temperature: t.Optional[float] = None,
        stop: t.Optional[t.List[str]] = None,
        callbacks: Callbacks = [],
    ) -> OutputModel:
        prompt_value = PromptValue(prompt_str=self.to_string(data))
        resp = await llm.generate(
            prompt_value,
            n=n,
            temperature=temperature,
            stop=stop,
            callbacks=callbacks,
        )
        resp_text = resp.generations[0][0].text
        parser = RagasoutputParser(pydantic_object=self.output_model)
        answer = await parser.aparse(resp_text, prompt_value, llm, max_retries=3)

        # TODO: make sure RagasOutputPraser returns the same type as OutputModel
        return answer  # type: ignore


class StringPrompt(BasePrompt):
    async def generate(
        self,
        llm: BaseRagasLLM,
        data: str,
        n: int = 1,
        temperature: t.Optional[float] = None,
        stop: t.Optional[t.List[str]] = None,
        callbacks: Callbacks = [],
    ) -> str:
        prompt_value = PromptValue(prompt_str=data)
        llm_result = await llm.agenerate_text(
            prompt_value,
            n=n,
            temperature=temperature,
            stop=stop,
            callbacks=callbacks,
        )
        return llm_result.generations[0][0].text

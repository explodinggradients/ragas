from __future__ import annotations

import inspect
import typing as t
from abc import ABC, abstractmethod

from pydantic import BaseModel

from ragas.llms.output_parser import RagasoutputParser
from ragas.llms.prompt import PromptValue

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks

    from ragas.llms.base import BaseRagasLLM


class BasePrompt(ABC):
    def __init__(self, name: t.Optional[str] = None):
        if name is None:
            self.name = self.__class__.__name__.lower()

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


InputModel = t.TypeVar("InputModel", bound=BaseModel)
OutputModel = t.TypeVar("OutputModel", bound=BaseModel)


class StringIO(BaseModel):
    text: str


class BoolIO(BaseModel):
    value: bool


class PydanticPrompt(BasePrompt, t.Generic[InputModel, OutputModel]):
    input_model: t.Type[InputModel]
    output_model: t.Type[OutputModel]
    instruction: str
    examples: t.List[t.Tuple[InputModel, OutputModel]] = []

    def generate_instruction(self) -> str:
        return self.instruction

    def generate_output_signature(self, indent: int = 4) -> str:
        return (
            f"Please return the output in a JSON format that complies with the "
            f"following schema as specified in JSON Schema and OpenAPI specification:\n"
            f"{self.output_model.model_json_schema()}"
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
                    + input_data.model_dump_json(indent=4)
                    + "\n"
                    + "output: "
                    + output_data.model_dump_json(indent=4)
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
            + data.model_dump_json(indent=4)
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
        processed_data = self.process_input(data)
        prompt_value = PromptValue(prompt_str=self.to_string(processed_data))
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
        return self.process_output(answer, data)  # type: ignore

    def process_input(self, input: InputModel) -> InputModel:
        return input

    def process_output(self, output: OutputModel, input: InputModel) -> OutputModel:
        return output


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


class PromptMixin:
    def get_prompts(self) -> t.Dict[str, PydanticPrompt]:
        prompts = {}
        for name, value in inspect.getmembers(self):
            if isinstance(value, PydanticPrompt):
                prompts.update({name: value})
        return prompts

    def set_prompts(self, **prompts):
        available_prompts = self.get_prompts()
        for key, value in prompts.items():
            if key not in available_prompts:
                raise ValueError(
                    f"Prompt with name '{key}' does not exist. Use get_prompts() to see available prompts."
                )
            if not isinstance(value, PydanticPrompt):
                raise ValueError(
                    f"Prompt with name '{key}' must be an instance of 'ragas.prompt.PydanticPrompt'"
                )
            setattr(self, key, value)

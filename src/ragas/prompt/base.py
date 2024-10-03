from __future__ import annotations

import typing as t
from abc import ABC, abstractmethod

from pydantic import BaseModel

from ragas.llms.output_parser import RagasOutputParser
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
        temperature: t.Optional[float] = None,
        stop: t.Optional[t.List[str]] = None,
        callbacks: Callbacks = [],
    ) -> t.Any:
        """
        Generate a single completion from the prompt.
        """
        pass

    @abstractmethod
    def generate_multiple(
        self,
        llm: BaseRagasLLM,
        data: t.Any,
        n: int = 1,
        temperature: t.Optional[float] = None,
        stop: t.Optional[t.List[str]] = None,
        callbacks: Callbacks = [],
    ) -> t.Any:
        """
        Generate multiple completions from the prompt.
        """
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

    def _generate_instruction(self) -> str:
        return self.instruction

    def _generate_output_signature(self, indent: int = 4) -> str:
        return (
            f"Please return the output in a JSON format that complies with the "
            f"following schema as specified in JSON Schema and OpenAPI specification:\n"
            f"{self.output_model.model_json_schema()}"
        )

    def _generate_examples(self):
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
            self._generate_instruction()
            + "\n"
            + self._generate_output_signature()
            + "\n"
            + self._generate_examples()
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
        temperature: t.Optional[float] = None,
        stop: t.Optional[t.List[str]] = None,
        callbacks: Callbacks = [],
    ) -> OutputModel:
        processed_data = self.process_input(data)
        prompt_value = PromptValue(prompt_str=self.to_string(processed_data))
        resp = await llm.generate(
            prompt_value,
            n=1,
            temperature=temperature,
            stop=stop,
            callbacks=callbacks,
        )
        resp_text = resp.generations[0][0].text
        parser = RagasOutputParser(pydantic_object=self.output_model)
        answer = await parser.aparse(resp_text, prompt_value, llm, max_retries=3)

        # TODO: make sure RagasOutputPraser returns the same type as OutputModel
        return self.process_output(answer, data)  # type: ignore

    async def generate_multiple(
        self,
        llm: BaseRagasLLM,
        data: InputModel,
        n: int = 1,
        temperature: t.Optional[float] = None,
        stop: t.Optional[t.List[str]] = None,
        callbacks: Callbacks = [],
    ) -> t.List[OutputModel]:
        processed_data = self.process_input(data)
        prompt_value = PromptValue(prompt_str=self.to_string(processed_data))
        resp = await llm.generate(
            prompt_value,
            n=n,
            temperature=temperature,
            stop=stop,
            callbacks=callbacks,
        )

        output_models = []
        parser = RagasOutputParser(pydantic_object=self.output_model)
        for i in range(n):
            resp_text = resp.generations[0][i].text
            answer = await parser.aparse(resp_text, prompt_value, llm, max_retries=3)
            output_models.append(self.process_output(answer, data))  # type: ignore

        return output_models

    def process_input(self, input: InputModel) -> InputModel:
        return input

    def process_output(self, output: OutputModel, input: InputModel) -> OutputModel:
        return output


class StringPrompt(BasePrompt):
    """
    A simple prompt that can be formatted with additional data using f-string syntax.

    This prompt is a simpler alternative to PydanticPrompt for those who prefer a more
    flexible approach without the need for a Pydantic model.

    Parameters
    ----------
    instruction : str
        The instruction string that can be formatted with additional data.

    Examples
    --------
    >>> prompt = StringPrompt(instruction="Generate a joke for the given category: {category}.")
    >>> await prompt.generate(llm=llm, data={"category": "commerce"})
    """

    instruction: str

    async def generate(
        self,
        llm: BaseRagasLLM,
        data: t.Optional[t.Dict[str, t.Any]] = None,
        temperature: t.Optional[float] = None,
        stop: t.Optional[t.List[str]] = None,
        callbacks: Callbacks = [],
    ) -> str:
        """
        Generate text based on the instruction and provided data.

        Parameters
        ----------
        llm : BaseRagasLLM
            The language model to use for text generation.
        data : Optional[Dict[str, Any]], optional
            The data to format the instruction with, by default None.
        n : int, optional
            The number of completions to generate, by default 1.
        temperature : Optional[float], optional
            The temperature for text generation, by default None.
        stop : Optional[List[str]], optional
            The stop sequences for text generation, by default None.
        callbacks : Callbacks, optional
            The callbacks to use during text generation, by default [].

        Returns
        -------
        str
            The generated text.
        """
        if data is None:
            data = {}
        prompt_value = PromptValue(prompt_str=self.instruction.format(**data))
        llm_result = await llm.agenerate_text(
            prompt_value,
            n=1,
            temperature=temperature,
            stop=stop,
            callbacks=callbacks,
        )
        return llm_result.generations[0][0].text

    async def generate_multiple(
        self,
        llm: BaseRagasLLM,
        data: t.Optional[t.Dict[str, t.Any]] = None,
        n: int = 1,
        temperature: t.Optional[float] = None,
        stop: t.Optional[t.List[str]] = None,
        callbacks: Callbacks = [],
    ) -> t.List[str]:
        return [
            await self.generate(llm, data, temperature, stop, callbacks)
            for _ in range(n)
        ]

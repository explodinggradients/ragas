from __future__ import annotations

import logging
import typing as t
from abc import ABC, abstractmethod

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel

from ragas.callbacks import new_group
from ragas.exceptions import RagasOutputParserException
from ragas.llms.prompt import PromptValue
from ragas.utils import RAGAS_SUPPORTED_LANGUAGE_CODES, camel_to_snake

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks

    from ragas.llms.base import BaseRagasLLM

logger = logging.getLogger(__name__)


class BasePrompt(ABC):
    def __init__(self, name: t.Optional[str] = None, language: str = "english"):
        if name is None:
            self.name = camel_to_snake(self.__class__.__name__)

        if language not in RAGAS_SUPPORTED_LANGUAGE_CODES:
            raise ValueError(
                f"Language '{language}' not supported. Supported languages: {RAGAS_SUPPORTED_LANGUAGE_CODES.keys()}"
            )
        self.language = language

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
        callbacks: t.Optional[Callbacks] = None,
    ) -> OutputModel:
        """
        Generate a single output using the provided language model and input data.

        This method is a special case of `generate_multiple` where only one output is generated.

        Parameters
        ----------
        llm : BaseRagasLLM
            The language model to use for generation.
        data : InputModel
            The input data for generation.
        temperature : float, optional
            The temperature parameter for controlling randomness in generation.
        stop : List[str], optional
            A list of stop sequences to end generation.
        callbacks : Callbacks, optional
            Callback functions to be called during the generation process.

        Returns
        -------
        OutputModel
            The generated output.

        Notes
        -----
        This method internally calls `generate_multiple` with `n=1` and returns the first (and only) result.
        """
        callbacks = callbacks or []

        # this is just a special case of generate_multiple
        output_single = await self.generate_multiple(
            llm=llm,
            data=data,
            n=1,
            temperature=temperature,
            stop=stop,
            callbacks=callbacks,
        )
        return output_single[0]

    async def generate_multiple(
        self,
        llm: BaseRagasLLM,
        data: InputModel,
        n: int = 1,
        temperature: t.Optional[float] = None,
        stop: t.Optional[t.List[str]] = None,
        callbacks: t.Optional[Callbacks] = None,
    ) -> t.List[OutputModel]:
        """
        Generate multiple outputs using the provided language model and input data.

        Parameters
        ----------
        llm : BaseRagasLLM
            The language model to use for generation.
        data : InputModel
            The input data for generation.
        n : int, optional
            The number of outputs to generate. Default is 1.
        temperature : float, optional
            The temperature parameter for controlling randomness in generation.
        stop : List[str], optional
            A list of stop sequences to end generation.
        callbacks : Callbacks, optional
            Callback functions to be called during the generation process.

        Returns
        -------
        List[OutputModel]
            A list of generated outputs.

        Raises
        ------
        RagasOutputParserException
            If there's an error parsing the output.
        """
        callbacks = callbacks or []
        processed_data = self.process_input(data)
        prompt_rm, prompt_cb = new_group(
            name=self.name,
            inputs={"data": processed_data},
            callbacks=callbacks,
        )
        prompt_value = PromptValue(prompt_str=self.to_string(processed_data))
        resp = await llm.generate(
            prompt_value,
            n=n,
            temperature=temperature,
            stop=stop,
            callbacks=prompt_cb,
        )

        output_models = []
        parser = RagasOutputParser(pydantic_object=self.output_model)
        for i in range(n):
            output_string = resp.generations[0][i].text
            try:
                answer = await parser.parse_output_string(
                    output_string=output_string,
                    prompt_value=prompt_value,
                    llm=llm,
                    callbacks=prompt_cb,
                    max_retries=3,
                )
                processed_output = self.process_output(answer, data)  # type: ignore
                output_models.append(processed_output)
            except RagasOutputParserException as e:
                prompt_rm.on_chain_error(error=e)
                logger.error("Prompt %s failed to parse output: %s", self.name, e)
                raise e

        prompt_rm.on_chain_end({"output": output_models})
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
    >>> from ragas.prompt import string_prompt
    >>> await prompt.generate(llm=llm, data={"category": "commerce"})
    """

    async def generate(
        self,
        llm: BaseRagasLLM,
        data: str,
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
        llm_result = await llm.agenerate_text(
            PromptValue(prompt_str=data),
            n=1,
            temperature=temperature,
            stop=stop,
            callbacks=callbacks,
        )
        return llm_result.generations[0][0].text

    async def generate_multiple(
        self,
        llm: BaseRagasLLM,
        data: str,
        n: int = 1,
        temperature: t.Optional[float] = None,
        stop: t.Optional[t.List[str]] = None,
        callbacks: Callbacks = [],
    ) -> t.List[str]:
        return [
            await self.generate(llm, data, temperature, stop, callbacks)
            for _ in range(n)
        ]


class OutputStringAndPrompt(BaseModel):
    output_string: str
    prompt_value: str


class FixOutputFormat(PydanticPrompt[OutputStringAndPrompt, StringIO]):
    instruction = "The output string did not satisfy the constraints given in the prompt. Fix the output string and return it."
    input_model = OutputStringAndPrompt
    output_model = StringIO


fix_output_format_prompt = FixOutputFormat()


class RagasOutputParser(PydanticOutputParser[OutputModel]):
    async def parse_output_string(
        self,
        output_string: str,
        prompt_value: PromptValue,
        llm: BaseRagasLLM,
        callbacks: Callbacks,
        max_retries: int = 1,
    ):
        callbacks = callbacks or []
        try:
            result = super().parse(output_string)
        except OutputParserException:
            if max_retries != 0:
                retry_rm, retry_cb = new_group(
                    name="fix_output_format",
                    inputs={"output_string": output_string},
                    callbacks=callbacks,
                )
                fixed_output_string = await fix_output_format_prompt.generate(
                    llm=llm,
                    data=OutputStringAndPrompt(
                        output_string=output_string,
                        prompt_value=prompt_value.to_string(),
                    ),
                )
                retry_rm.on_chain_end({"fixed_output_string": fixed_output_string})
                return await self.parse_output_string(
                    output_string=fixed_output_string.text,
                    prompt_value=prompt_value,
                    llm=llm,
                    max_retries=max_retries - 1,
                    callbacks=callbacks,
                )
            else:
                raise RagasOutputParserException(num_retries=max_retries)
        return result

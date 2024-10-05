from __future__ import annotations

import copy
import logging
import typing as t

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel

from ragas.callbacks import new_group
from ragas.exceptions import RagasOutputParserException
from ragas.llms.prompt import PromptValue

from .base import BasePrompt, StringIO, _check_if_language_is_supported
from .utils import get_all_strings, update_strings

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks

    from ragas.llms.base import BaseRagasLLM

logger = logging.getLogger(__name__)

# type variables for input and output models
InputModel = t.TypeVar("InputModel", bound=BaseModel)
OutputModel = t.TypeVar("OutputModel", bound=BaseModel)


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

    async def adapt(
        self, target_language: str, llm: BaseRagasLLM
    ) -> "PydanticPrompt[InputModel, OutputModel]":
        """
        Adapt the prompt to a new language.
        """

        # throws ValueError if language is not supported
        _check_if_language_is_supported(target_language)

        strings = get_all_strings(self.examples)
        translated_strings = await translate_statements_prompt.generate(
            llm=llm,
            data=ToTranslate(target_language=target_language, statements=strings),
        )

        translated_examples = update_strings(
            obj=self.examples,
            old_strings=strings,
            new_strings=translated_strings.statements,
        )

        new_prompt = copy.deepcopy(self)
        new_prompt.examples = translated_examples
        new_prompt.language = target_language
        return new_prompt


# Ragas Output Parser
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


# Ragas Adaptation
class ToTranslate(BaseModel):
    target_language: str
    statements: t.List[str]


class Translated(BaseModel):
    statements: t.List[str]


class TranslateStatements(PydanticPrompt[ToTranslate, Translated]):
    instruction = "Translate the following statements to the target language."
    input_model = ToTranslate
    output_model = Translated
    examples = [
        (
            ToTranslate(
                target_language="hindi",
                statements=[
                    "Albert Einstein was born in Germany.",
                    "Albert Einstein was best known for his theory of relativity.",
                ],
            ),
            Translated(
                statements=[
                    "अल्बर्ट आइंस्टीन का जन्म जर्मनी में हुआ था।",
                    "अल्बर्ट आइंस्टीन अपने सापेक्षता के सिद्धांत के लिए सबसे अधिक प्रसिद्ध थे।",
                ]
            ),
        ),
        (
            ToTranslate(
                target_language="dutch",
                statements=[
                    "Paris is the capital of France.",
                    "Croissants are a popular French pastry.",
                ],
            ),
            Translated(
                statements=[
                    "Parijs is de hoofdstad van Frankrijk.",
                    "Croissants zijn een populair Frans gebak.",
                ]
            ),
        ),
    ]

    def process_output(self, output: Translated, input: ToTranslate) -> Translated:
        if len(output.statements) != len(input.statements):
            raise ValueError(
                "The number of statements in the output does not match the number of statements in the input. Translation failed."
            )
        return output


translate_statements_prompt = TranslateStatements()

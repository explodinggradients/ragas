import json
import logging
import typing as t

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel

from ragas.llms import BaseRagasLLM
from ragas.llms.prompt import Prompt, PromptValue

logger = logging.getLogger(__name__)
# The get_format_instructions function is a modified version from
# langchain_core.output_parser.pydantic. The original version removed the "type" json schema
# property that confused some older LLMs.

TBaseModel = t.TypeVar("TBaseModel", bound=BaseModel)

FIX_OUTPUT_FORMAT = Prompt(
    name="",
    instruction="Below, the Completion did not satisfy the constraints given in the Prompt.",
    output_format_instruction="",
    input_keys=["prompt", "completion"],
    output_key="fixed_completion",
)


JSON_FORMAT_INSTRUCTIONS = """The output should be a well-formatted JSON instance that conforms to the JSON schema below.

As an example, for the schema {{"properties": {{"foo": {{"title": "Foo", "description": "a list of strings", "type": "array", "items": {{"type": "string"}}}}}}, "required": ["foo"]}}
the object {{"foo": ["bar", "baz"]}} is a well-formatted instance of the schema. The object {{"properties": {{"foo": ["bar", "baz"]}}}} is not well-formatted.

Here is the output JSON schema:
```
{schema}
```

Do not return any preamble or explanations, return only a pure JSON string surrounded by triple backticks (```)."""


def get_json_format_instructions(pydantic_object: t.Type[TBaseModel]) -> str:
    # Copy schema to avoid altering original Pydantic schema.
    schema = {k: v for k, v in pydantic_object.schema().items()}

    # Remove extraneous fields.
    reduced_schema = schema
    if "title" in reduced_schema:
        del reduced_schema["title"]
    # Ensure json in context is well-formed with double quotes.
    schema_str = json.dumps(reduced_schema)

    resp = JSON_FORMAT_INSTRUCTIONS.format(schema=schema_str)
    return resp


class RagasoutputParser(PydanticOutputParser):
    async def aparse(  # type: ignore
        self, result: str, prompt: PromptValue, llm: BaseRagasLLM, max_retries: int = 1
    ):
        try:
            output = super().parse(result)
        except OutputParserException:
            if max_retries != 0:
                p_value = FIX_OUTPUT_FORMAT.format(
                    prompt=prompt.to_string(), completion=result
                )
                output = await llm.generate(p_value)
                result = output.generations[0][0].text
                return await self.aparse(result, prompt, llm, max_retries - 1)
            else:
                logger.warning("Failed to parse output. Returning None.")
                return None
        return output

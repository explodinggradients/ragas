import typing as t
import json

from langchain_core.pydantic_v1 import BaseModel

# The get_format_instructions function is a modified version from
# langchain_core.output_parser.pydantic. The original version removed the "type" json schema
# property that confused some older LLMs.

TBaseModel = t.TypeVar("TBaseModel", bound=BaseModel)

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

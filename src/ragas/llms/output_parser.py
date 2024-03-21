import typing as t
import json

from langchain_core.pydantic_v1 import BaseModel
from langchain_core.output_parsers.format_instructions import JSON_FORMAT_INSTRUCTIONS

# The get_format_instructions function is a modified version from
# langchain_core.output_parser.pydantic. The original version removed the "type" json schema
# property that confused some older LLMs.

TBaseModel = t.TypeVar("TBaseModel", bound=BaseModel)

def get_json_format_instructions(pydantic_object: t.Type[TBaseModel]) -> str:
    # Copy schema to avoid altering original Pydantic schema.
    schema = {k: v for k, v in pydantic_object.schema().items()}

    # Remove extraneous fields.
    reduced_schema = schema
    if "title" in reduced_schema:
        del reduced_schema["title"]
    # Ensure json in context is well-formed with double quotes.
    schema_str = json.dumps(reduced_schema)

    return JSON_FORMAT_INSTRUCTIONS.format(schema=schema_str)

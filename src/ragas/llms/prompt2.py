import typing as t

from langchain.prompts import BasePromptTemplate, PromptTemplate, StringPromptTemplate
from langchain_core.output_parsers.base import BaseOutputParser
from langchain_core.output_parsers.json import JsonOutputParser
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain.pydantic_v1 import Field

from ragas.llms.output_parser import get_json_format_instructions


class ExamplePromptTemplate(StringPromptTemplate):

    input_variables: t.List[str] = Field(default_factory=lambda: ["key", "value"])
    """A list of the names of the variables the prompt template expects.
    Defaults to ["key", "value"]."""

    template: str = "{key}: {value}"
    """The prompt template to format the examples. Defaults to "{key}: {value}\\n"."""

    element_separator: str = "\n"
    """The separator between the prompt elements."""

    def format(self, **kwargs: t.Any) -> str:
        return self.element_separator.join(
            self.template.format(key=k, value=v) for k, v in kwargs.items()
        )


def get_prompt(
    instructions: str,
    output_parser: t.Optional[BaseOutputParser] = None,
    examples: t.List[t.Dict[str, t.Any]] = [],
    example_prompt: BasePromptTemplate = ExamplePromptTemplate(),
    example_separator: str = "\n\n",
    input_variables: t.List[str] = [],
    output_key: str = "output",
) -> PromptTemplate:
    template = "{}".format(instructions)

    if output_parser:
        if (
            isinstance(output_parser, (JsonOutputParser, PydanticOutputParser))
            and output_parser.pydantic_object
        ):
            output_instructions = get_json_format_instructions(output_parser.pydantic_object)
        else:
            output_instructions = output_parser.get_format_instructions()
        output_instructions = output_instructions.replace("{", "{{").replace("}", "}}")
        template += "\n\n{}".format(output_instructions)

    if examples:
        example_str = example_separator.join(
            example_prompt.format(**example) for example in examples
        )
        example_str = example_str.replace("{", "{{").replace("}", "}}")
        template += "\n\nExamples:\n\n{}".format(example_str)

    task_elements = { input_key: f"{{{input_key}}}" for input_key in input_variables }
    task_elements[output_key] = ""
    task_description = example_prompt.format(**task_elements)
    template += "\n\nYour actual task:\n\n{}".format(task_description)

    return PromptTemplate(
        input_variables=input_variables,
        template=template,
    )

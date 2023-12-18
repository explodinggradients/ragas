from __future__ import annotations

import json
import typing as t

from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.prompt_values import PromptValue
from langchain_core.messages import BaseMessage, HumanMessage

try:
    from pydantic.v1 import root_validator
except ImportError:
    from pydantic import root_validator

class Prompt(PromptValue):
    """
    RagasPrompt is a class that represents a prompt for the ragas metrics.
    """
    instruction: str
    examples: t.List[t.Dict[str, t.Any]] = []
    input_keys: t.List[str]
    output_key: str
    output_type: str = 'json'

    @root_validator
    def validate_prompt(cls, value: t.Dict[str, t.Any]) -> t.Dict[str, t.Any]:
        """
        Validate the template string to ensure that it is in desired format.
        """
        if value.get("instruction") is None or value.get("instruction") == "":
            raise ValueError(
                "instruction cannot be empty"
            )
        if value.get("input_keys") is None or value.get("instruction") == []:
            raise ValueError(
                "input_keys cannot be empty"
            )
        if value.get("output_key") is None or value.get("output_key") == "":
            raise ValueError(
                "output_key cannot be empty"
            )
        
        if value.get("examples"):
            output_key = value["output_key"]
            for no, example in enumerate(value['examples']):
                for inp_key in value['input_keys']:
                    if inp_key not in example:
                        raise ValueError(
                            f"example {no+1} does not have the variable {inp_key} in the definition"
                        )
                if output_key not in example:
                    raise ValueError(
                        f"example {no+1} does not have the variable {output_key} in the definition"
                    )
                if value["output_type"] == 'json':
                    try:
                        if output_key in example:
                            json.loads(example[output_key])
                    except ValueError as e:
                        raise ValueError(
                            f"{output_key} in example {no+1} is not in valid json format: {e}"
                        )

        return value

    def to_string(self) -> str:
        """
        Generate the prompt string from the variables.
        """
        prompt_str = self.instruction + '\n'

        # Format the examples to match the Langchain prompt template
        for example in self.examples:
            for key, value in example.items():
                value = value.replace('{','{{').replace('}','}}') if self.output_type == 'json' else value
                prompt_str += f'\n{key}: {value}'
            prompt_str += '\n'
        
        prompt_str += ''.join(f'\n{key}: {{{key}}}' for key in self.input_keys)
        prompt_str += f'\n{self.output_key}: \n'

        return prompt_str

    def to_messages(self) -> t.List[BaseMessage]:
        """Return prompt as a list of Messages."""
        return [HumanMessage(content=self.to_string())]
    
    def get_example_str(self, example_no: int) -> str:
        """
        Get the example string from the example number.
        """
        if example_no >= len(self.examples):
            raise ValueError(
                f"example number {example_no} is out of range"
            )
        example = self.examples[example_no]
        example_str = ''
        for key, value in example.items():
            value = value.replace('{','{{').replace('}','}}') if self.output_type == 'json' else value
            example_str += f'\n{key}: {value}'
        return example_str

    def format(self, **kwargs: t.Any) -> ChatPromptTemplate:
        """
        Format the RagasPrompt object into a ChatPromptTemplate object to be used in metrics.
        """
        if set(self.input_keys) != set(kwargs.keys()):
            raise ValueError(
                f"Input variables {self.input_keys} do not match with the given parameters {list(kwargs.keys())}"
            )
        prompt = self.to_string()
        human_prompt = HumanMessagePromptTemplate.from_template(prompt)
        return ChatPromptTemplate.from_messages([human_prompt.format(**kwargs)])

from __future__ import annotations

import json
import typing as t

from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.prompt_values import PromptValue
from langchain_core.messages import BaseMessage

try:
    from pydantic.v1 import root_validator
except ImportError:
    from pydantic import root_validator

class RagasPrompt(PromptValue):
    """
    RagasPrompt is a class that represents a prompt for the ragas metrics.
    """
    instruction: str
    examples: t.Optional[t.List[dict[str, t.Any]]] = []
    input_keys: t.List[str]
    output_key: str
    output_type: t.Optional[str] = 'JSON'
    
    def to_string(self) -> str:
        """Return prompt value as string."""
        ...
    
    def to_messages(self) -> t.List[BaseMessage]:
        """Return prompt as a list of Messages."""
        ...

    @root_validator
    def validate(cls, values: t.Dict) -> t.Dict:
        """
        Validate the template string to ensure that it is in desired format.
        """
        if values.get("instruction") is None or values.get("instruction") == "":
            raise ValueError(
                "Instruction cannot be empty"
            )
        if values.get("input_keys") is None or values.get("instruction") == []:
            raise ValueError(
                "Input keys cannot be empty"
            )
        if values.get("output_key") is None or values.get("output_key") == "":
            raise ValueError(
                "Output key cannot be empty"
            )
        
        if values.get("examples"):
            output_key = values.get("output_key")
            for no, example in enumerate(values.get("examples")):
                for inp_key in values.get("input_keys"):
                    if inp_key not in example:
                        raise ValueError(
                            f"Example {no+1} does not have the variable {inp_key} in the definition"
                        )
                if output_key not in example:
                    raise ValueError(
                        f"Example {no+1} does not have the variable {output_key} in the definition"
                    )
                if values.get("output_type") == 'JSON':
                    try:
                        json.loads(example[output_key])
                    except ValueError as e:
                        raise ValueError(
                            f"{output_key} in example {no+1} is not in valid JSON format: {e}"
                        )

        return values

    def generate_prompt_string(self) -> str:
        """
        Generate the prompt string from the variables.
        """
        prompt_str = self.instruction + '\n'

        # Format the examples to match the Langchain prompt template
        for example in self.examples:
            for key, value in example.items():
                value = value.replace('{','{{').replace('}','}}') if self.output_type == 'JSON' else value
                prompt_str += f'\n{key}: {value}'
            prompt_str += '\n'
        
        prompt_str += ''.join(f'\n{key}: {{{key}}}' for key in self.input_keys)
        prompt_str += f'\n{self.output_key}: \n'

        return prompt_str

    def format(self, **kwargs: t.Any) -> str:
        """
        Format the RagasPrompt object into a ChatPromptTemplate object to be used in metrics.
        """
        if set(self.input_keys) != set(kwargs.keys()):
            raise ValueError(
                f"Input variables {self.input_keys} do not match with the given parameters {list(kwargs.keys())}"
            )
        prompt = self.generate_prompt_string()
        human_prompt = HumanMessagePromptTemplate.from_template(prompt)
        return ChatPromptTemplate.from_messages([human_prompt.format(**kwargs)])

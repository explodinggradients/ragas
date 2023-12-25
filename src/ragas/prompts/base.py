from __future__ import annotations

import json
import typing as t

from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import BaseMessage
from langchain_core.prompt_values import PromptValue
import pickle
import os

from ragas.llms.base import RagasLLM
from ragas.utils import json_loader

try:
    from pydantic.v1 import root_validator
except ImportError:
    from pydantic import root_validator


class RagasPrompt(PromptValue):
    """
    RagasPrompt is a class that represents a prompt for the ragas metrics.
    """
    name: str
    instruction: str
    examples: t.List[t.Dict[str, t.Any]] = []
    input_keys: t.List[str]
    output_key: str
    output_type: str = "JSON"
    language: str = "en"

    def to_string(self) -> str:
        """Return prompt value as string."""
        ...

    def to_messages(self) -> t.List[BaseMessage]:
        """Return prompt as a list of Messages."""
        ...

    @root_validator()
    def validate_prompt(cls, value: t.Dict[str, str]) -> t.Dict[str, str]:
        """
        Validate the template string to ensure that it is in desired format.
        """
        if value.get("instruction") is None or value.get("instruction") == "":
            raise ValueError("Instruction cannot be empty")
        if value.get("input_keys") is None or value.get("instruction") == []:
            raise ValueError("Input keys cannot be empty")
        if value.get("output_key") is None or value.get("output_key") == "":
            raise ValueError("Output key cannot be empty")

        if value.get("examples"):
            output_key = value["output_key"]
            for no, example in enumerate(value["examples"]):
                for inp_key in value["input_keys"]:
                    if inp_key not in example:
                        raise ValueError(
                            f"Example {no+1} does not have the variable {inp_key} in the definition"
                        )
                if output_key not in example:
                    raise ValueError(
                        f"Example {no+1} does not have the variable {output_key} in the definition"
                    )
                if value["output_type"] == "JSON":
                    try:
                        json.loads(example[output_key])
                    except ValueError as e:
                        raise ValueError(
                            f"{output_key} in example {no+1} is not in valid JSON format: {e}"
                        )

        return value

    def generate_prompt_string(self) -> str:
        """
        Generate the prompt string from the variables.
        """
        prompt_str = self.instruction + "\n"

        # Format the examples to match the Langchain prompt template
        for example in self.examples:
            for key, value in example.items():
                value = (
                    value.replace("{", "{{").replace("}", "}}")
                    if self.output_type == "JSON"
                    else value
                )
                prompt_str += f"\n{key}: {value}"
            prompt_str += "\n"

        prompt_str += "".join(f"\n{key}: {{{key}}}" for key in self.input_keys)
        prompt_str += f"\n{self.output_key}: \n"

        return prompt_str

    def format(self, **kwargs: t.Any) -> ChatPromptTemplate:
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

    def adapt(self, language: str, cache_dir: str, llm: RagasLLM) -> None:
        # TODO: Add callbacks
        cache_dir = cache_dir if cache_dir else "~/.cache/ragas/prompts"
        if os.path.exists(os.path.join(cache_dir, language, f"{self.name}.json")):
            self._load(language, self.name, cache_dir)
        
        prompts = []
        for example in self.examples:
            prompts.extend(
                [
                    str_translation.format(translate_to=language, input=example.get(key))
                    for key in self.input_keys
                ]
            )
            prompts.append(
                json_translatation.format(
                    translate_to=language, input=example.get(self.output_key)
                )
                if self.output_type.lower() == "json"
                else str_translation.format(translate_to=language, input=example.get(self.output_key))
            )

        results = [result[0].text for result in llm.generate(prompts).generations]
        per_example_items = len(self.input_keys) + 1
        grouped_results = [
            results[i : i + per_example_items]
            for i in range(0, len(results), per_example_items)
        ]
        assert len(grouped_results) == len(
            self.examples
        ), "examples and adapted examples must be of equal length"
        for i, example in enumerate(grouped_results):
            example_dict = {}
            example_dict.update(
                {k: v for k, v in zip(self.input_keys, example[: len(self.input_keys)])}
            )
            example_dict[self.output_key] = (
                json_loader.safe_load(example[-1], llm=llm)
                if self.output_type.lower() == "json"
                else example[-1]
            )
            self.examples[i] = example_dict
            
        self.language = language
        
    def save(self, cache_dir: str = "~/.cache/ragas/prompts") -> None:
        
        cache_dir = os.path.join(cache_dir, self.language)
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        
        cache_path = os.path.join(cache_dir, f"{self.name}.json")
        self.to_json(cache_path)
      

    @classmethod 
    def _load(cls, language: str, name: str, cache_dir: str ) -> RagasPrompt:
        
        path = os.path.join(cache_dir, language, f"{name}.json")
        cls(**json.load(open(path)))

str_translation = RagasPrompt(
    name="str_translation",
    instruction="Language translation",
    examples=[
        {
            "translate_to": "hindi",
            "input": "Who was  Albert Einstein and what is he best known for?",
            "output": "अल्बर्ट आइंस्टीन कौन थे और वे किसके लिए सबसे ज्यादा प्रसिद्ध हैं?",
        },
    ],
    input_keys=["translate_to", "input"],
    output_key="output",
    output_type="str",
)

json_translatation = RagasPrompt(
    name="json_translation",
    instruction="Translate values in given json to target language ",
    examples=[
        {
            "translate_to": "hindi",
            "input": """{
        "statements": [
            "Albert Einstein was born in Germany.",
            "Albert Einstein was best known for his theory of relativity."
        ]}""",
            "output": """{
    "statements": [
    "अल्बर्ट आइंस्टीन का जन्म जर्मनी में हुआ था।",
    "अल्बर्ट आइंस्टीन अपने सापेक्षता के सिद्धांत के लिए सबसे अधिक प्रसिद्ध थे।"
    ]}""",
        }
    ],
    input_keys=["translate_to", "input"],
    output_key="output",
    output_type="JSON",
)

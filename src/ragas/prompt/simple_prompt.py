from __future__ import annotations

__all__ = ["Prompt"]

import gzip
import json
import typing as t
import warnings
from pathlib import Path

if t.TYPE_CHECKING:
    from pydantic import BaseModel


class Prompt:
    def __init__(
        self,
        instruction: str,
        examples: t.Optional[t.List[t.Tuple[t.Dict, t.Dict]]] = None,
        response_model: t.Optional[BaseModel] = None,
    ):
        """
        Create a simple prompt object.

        Parameters:
        -----------
        instruction : str
            The prompt instruction template with placeholders like {response}, {expected_answer}
        examples : Optional[List[Tuple[Dict, Dict]]]
            List of (input_dict, output_dict) pairs for few-shot learning
        response_model: Optional[BaseModel]
            The expected response model

        Examples:
        ---------
        Basic prompt with placeholders:

        >>> prompt = Prompt("Answer the question: {question}")
        >>> formatted = prompt.format(question="What is 2+2?")
        >>> print(formatted)
        Answer the question: What is 2+2?

        Prompt with few-shot examples:

        >>> examples = [
        ...     ({"question": "What is 1+1?"}, {"answer": "2"}),
        ...     ({"question": "What is 3+3?"}, {"answer": "6"})
        ... ]
        >>> prompt = Prompt(
        ...     "Answer: {question}",
        ...     examples=examples
        ... )
        >>> formatted = prompt.format(question="What is 5+5?")
        >>> print(formatted)
        Answer: What is 5+5?

        Examples:
        Example 1:
        Input:
        question: What is 1+1?
        Output:
        answer: 2

        Example 2:
        Input:
        question: What is 3+3?
        Output:
        answer: 6

        Adding examples dynamically:

        >>> prompt = Prompt("Translate to {language}: {text}")
        >>> prompt.add_example(
        ...     {"text": "Hello", "language": "Spanish"},
        ...     {"translation": "Hola"}
        ... )
        >>> formatted = prompt.format(text="Goodbye", language="French")

        Save and load prompts:

        >>> prompt.save("my_prompt.json")
        >>> loaded_prompt = Prompt.load("my_prompt.json")
        >>> # With compression
        >>> prompt.save("compressed_prompt.json.gz")
        >>> loaded_compressed = Prompt.load("compressed_prompt.json.gz")
        """
        self.instruction = instruction
        self.response_model = response_model

        # Add examples if provided
        self.examples = []
        if examples:
            for inputs, output in examples:
                self.add_example(inputs, output)

    def format(self, **kwargs) -> str:
        """Format the prompt with the provided variables."""

        prompt_parts = []
        prompt_parts.append(self.instruction.format(**kwargs))
        if self.examples:
            prompt_parts.append(self._format_examples())

        # Combine all parts
        if len(prompt_parts) > 1:
            return "\n\n".join(prompt_parts)
        else:
            return prompt_parts[0]

    def _format_examples(self) -> str:
        # Add examples in a simple format
        examples = []
        if self.examples:
            examples.append("Examples:")
            for i, (inputs, output) in enumerate(self.examples, 1):
                example_input = "\n".join([f"{k}: {v}" for k, v in inputs.items()])
                example_output = "\n".join([f"{k}: {v}" for k, v in output.items()])

                examples.append(
                    f"Example {i}:\nInput:\n{example_input}\nOutput:\n{example_output}"
                )

        return "\n\n".join(examples) if examples else ""

    def add_example(self, input: t.Dict, output: t.Dict) -> None:
        """
        Add an example to the prompt.

        Parameters:
        -----------
        inputs : Dict
            Dictionary of input values
        output : Dict
            Dictionary of output values

        Raises:
        -------
        TypeError
            If inputs or output is not a dictionary
        """
        if not isinstance(input, dict):
            raise TypeError(f"Expected inputs to be dict, got {type(input).__name__}")
        if not isinstance(output, dict):
            raise TypeError(f"Expected output to be dict, got {type(output).__name__}")

        self.examples.append((input, output))

    def save(self, path: str) -> None:
        """
        Save the prompt to a JSON file.

        Parameters:
        -----------
        path : str
            File path to save to. Use .gz extension for compression.

        Note:
        -----
        If the prompt has a response_model, its schema will be saved for reference
        but the model itself cannot be serialized. You'll need to provide it when loading.
        """
        if self.response_model:
            warnings.warn(
                "response_model cannot be saved and will be lost. "
                "You'll need to set it manually after loading using: "
                "Prompt.load(path, response_model=YourModel)"
            )

        data = {
            "format_version": "1.0",
            "type": "Prompt",
            "instruction": self.instruction,
            "examples": [{"input": inp, "output": out} for inp, out in self.examples],
            "response_model_info": self._serialize_response_model_info(),
        }

        file_path = Path(path)
        try:
            if file_path.suffix == ".gz":
                with gzip.open(file_path, "wt", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
            else:
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
        except (OSError, IOError) as e:
            raise ValueError(f"Cannot save prompt to {path}: {e}")

    @classmethod
    def load(
        cls, path: str, response_model: t.Optional["BaseModel"] = None
    ) -> "Prompt":
        """
        Load a prompt from a JSON file.

        Parameters:
        -----------
        path : str
            File path to load from. Supports .gz compressed files.
        response_model : Optional[BaseModel]
            Pydantic model to use for response validation. Required if the
            original prompt had a response_model.

        Returns:
        --------
        Prompt
            Loaded prompt instance

        Raises:
        -------
        ValueError
            If file cannot be loaded, is invalid, or missing required response_model
        """
        file_path = Path(path)

        # Load JSON data
        try:
            if file_path.suffix == ".gz":
                with gzip.open(file_path, "rt", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, OSError) as e:
            raise ValueError(f"Cannot load prompt from {path}: {e}")

        # Validate format
        if data.get("type") != "Prompt":
            raise ValueError(
                f"File is not a Prompt (found type: {data.get('type', 'unknown')})"
            )

        # Check if response_model is required but not provided
        response_model_info = data.get("response_model_info")
        if response_model_info and not response_model:
            raise ValueError(
                f"This prompt requires a response_model of type '{response_model_info['class_name']}'\n"
                f"Usage: Prompt.load('{path}', response_model=YourModel)"
            )

        # Extract examples
        examples = [(ex["input"], ex["output"]) for ex in data.get("examples", [])]

        # Create prompt instance
        prompt = cls(
            instruction=data["instruction"],
            examples=examples,
            response_model=response_model,
        )

        # Validate response model if both provided and expected
        if response_model and response_model_info:
            prompt._validate_response_model(response_model, response_model_info)

        return prompt

    def _serialize_response_model_info(self) -> t.Optional[t.Dict]:
        """Serialize response model information for storage."""
        if not self.response_model:
            return None

        return {
            "class_name": self.response_model.__class__.__name__,
            "module": self.response_model.__class__.__module__,
            "schema": self.response_model.model_json_schema(),
            "note": "You must provide this model when loading",
        }

    def _validate_response_model(
        self, provided_model: "BaseModel", expected_info: t.Dict
    ) -> None:
        """Validate that provided response model matches expected schema."""
        if not provided_model:
            return

        expected_schema = expected_info.get("schema", {})
        actual_schema = provided_model.model_json_schema()

        # Compare key schema properties
        if expected_schema.get("properties") != actual_schema.get(
            "properties"
        ) or expected_schema.get("required") != actual_schema.get("required"):
            warnings.warn(
                f"Provided response_model schema differs from saved model "
                f"(expected: {expected_info['class_name']})"
            )

    def __str__(self) -> str:
        """String representation showing the instruction."""
        return f"Prompt(instruction='{self.instruction}', examples={self.examples}, response_model={self.response_model})"

    __repr__ = __str__

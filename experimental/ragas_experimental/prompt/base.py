from __future__ import annotations

__all__ = ["Prompt"]

import typing as t

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

    def __str__(self) -> str:
        """String representation showing the instruction."""
        return f"Prompt(instruction='{self.instruction}', examples={self.examples}, response_model={self.response_model})"

    __repr__ = __str__

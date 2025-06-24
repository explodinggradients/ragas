__all__ = ["Prompt"]

import re
import typing as t


class Prompt:
    def __init__(
        self,
        instruction: str,
        examples: t.Optional[t.List[t.Tuple[t.Dict, t.Dict]]] = None,
    ):
        """
        Create a simple prompt object.

        Parameters:
        -----------
        instruction : str
            The prompt instruction template with placeholders like {response}, {expected_answer}
        examples : Optional[List[Tuple[Dict, Dict]]]
            List of (input_dict, output_dict) pairs for few-shot learning
        """
        self.instruction = instruction
        self.examples = []

        # Validate the instruction
        self._validate_instruction()

        # Add examples if provided
        if examples:
            for inputs, output in examples:
                self.add_example(inputs, output)

    def _validate_instruction(self):
        """Ensure the instruction contains at least one placeholder."""
        if not re.findall(r"\{(\w+)\}", self.instruction):
            raise ValueError(
                "Instruction must contain at least one placeholder like {response}"
            )

    def format(self, **kwargs) -> str:
        """Format the prompt with the provided variables."""

        prompt_parts = []
        prompt_parts.append(self.instruction.format(**kwargs))
        prompt_parts.append(self._format_examples())

        # Combine all parts
        return "\n\n".join(prompt_parts)

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

    def add_example(self, inputs: t.Dict, output: t.Dict) -> None:
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
        if not isinstance(inputs, dict):
            raise TypeError(f"Expected inputs to be dict, got {type(inputs).__name__}")
        if not isinstance(output, dict):
            raise TypeError(f"Expected output to be dict, got {type(output).__name__}")

        self.examples.append((inputs, output))

    def __str__(self) -> str:
        """String representation showing the instruction."""
        return f"Prompt(instruction='{self.instruction}',\n examples={self.examples})"

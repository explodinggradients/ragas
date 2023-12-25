from __future__ import annotations

import json
import typing as t
from dataclasses import dataclass

from langchain.callbacks.manager import CallbackManager, trace_as_chain_group

from ragas.llms.prompt import Prompt

if t.TYPE_CHECKING:
    from ragas.llms import RagasLLM

JSON_PROMPT = Prompt(
    name="json_safeloader",
    instruction="Rewrite the input into valid json",
    examples=[
        {
            "input": """{
    "name": "John Doe",
    "age": 30,
    "isStudent": false
    "address": {
        "street": "123 Main St",
        "city": "Anytown",
        "state": "CA",
    }
    "hobbies": ["reading", "swimming", "cycling"]
    }""",
            "output": """{
        "name": "John Doe",
        "age": 30,
        "isStudent": false,
        "address": {
            "street": "123 Main St",
            "city": "Anytown",
            "state": "CA"
        },
        "hobbies": ["reading", "swimming", "cycling"]
    }""",
        },
        {
            "input": """{
        "statement": "The Earth is also known as "Terra" "
        }""",
            "output": """{
        "statement": "The Earth is also known as 'Terra'"
    }
""",
        },
    ],
    input_keys=["input"],
    output_key="output",
    output_type="JSON",
)


@dataclass
class JsonLoader:
    max_retries: int = 2

    def safe_load(self, text: str, llm: RagasLLM):
        retry = 0
        while retry <= self.max_retries:
            try:
                start, end = self._find_outermost_json(text)
                return json.loads(text[start:end])
            except ValueError:
                text = self._fix_to_json(text, llm)
            retry += 1

        return {}

    def _fix_to_json(
        self,
        text,
        llm,
        callbacks: t.Optional[CallbackManager] = None,
        callback_group_name: str = "batch",
    ):
        # TODO (executor)
        with trace_as_chain_group(
            callback_group_name, callback_manager=callbacks
        ) as batch_group:
            human_prompt = JSON_PROMPT.format(input=text)

            results = llm.generate(
                [human_prompt],
                n=1,
                callbacks=batch_group,
            )
        return results.generations[0][0].text

    def _find_outermost_json(self, text):
        stack = []
        start_index = -1

        for i, char in enumerate(text):
            if char in "{[":
                if len(stack) == 0:
                    start_index = i
                stack.append(char)

            elif char in "}]":
                if len(stack) > 0:
                    last = stack.pop()
                    if (char == "}" and last != "{") or (char == "]" and last != "["):
                        # Mismatched closing brace/bracket, invalid JSON
                        break

                if len(stack) == 0 and start_index != -1:
                    # Found a valid outermost JSON
                    return (
                        start_index,
                        i + 1,
                    )  # Add 1 to include the closing brace/bracket in the range

        return -1, -1  # No valid JSON found


json_loader = JsonLoader()

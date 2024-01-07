from __future__ import annotations

import json
import logging
import typing as t
from dataclasses import dataclass

logger = logging.getLogger(__name__)

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks

    from ragas.llms.base import BaseRagasLLM


def load_as_json(text) -> t.Dict:
    """
    validate and return given text as json
    """

    try:
        return json.loads(text)
    except ValueError as e:
        logger.warn(f"Invalid json: {e}")
        return {}


# not migrating to Prompt format to avoid circular imports
JSON_PROMPT = """\
Rewrite the input into valid json

Input:
{{
    "name": "John Doe",
    "age": 30,
    "isStudent": false
    "address": {{
        "street": "123 Main St",
        "city": "Anytown",
        "state": "CA",
    }}
    "hobbies": ["reading", "swimming", "cycling"]
}}
Output:
{{
    "name": "John Doe",
    "age": 30,
    "isStudent": false,
    "address": {{
        "street": "123 Main St",
        "city": "Anytown",
        "state": "CA"
    }},
    "hobbies": ["reading", "swimming", "cycling"]
}}


Input:
{{
    "statement": "The Earth is also known as "Terra" "
}}
Output:
{{
    "statement": "The Earth is also known as 'Terra'"
}}

Input:
{input}

Output:
"""


@dataclass
class JsonLoader:
    max_retries: int = 2

    def safe_load(self, text: str, llm: BaseRagasLLM, callbacks: Callbacks = None):
        retry = 0
        while retry <= self.max_retries:
            try:
                start, end = self._find_outermost_json(text)
                return json.loads(text[start:end])
            except ValueError:
                from ragas.llms.prompt import PromptValue

                results = llm.generate_text(
                    PromptValue(prompt_str=JSON_PROMPT.format(input=text)),
                    n=1,
                    callbacks=callbacks,
                )
                text = results.generations[0][0].text
            retry += 1

        return {}

    async def asafe_load(
        self, text: str, llm: BaseRagasLLM, callbacks: Callbacks = None
    ):
        retry = 0
        while retry <= self.max_retries:
            try:
                start, end = self._find_outermost_json(text)
                return json.loads(text[start:end])
            except ValueError:
                from ragas.llms.prompt import PromptValue

                results = await llm.agenerate_text(
                    PromptValue(prompt_str=JSON_PROMPT.format(input=text)),
                    n=1,
                    callbacks=callbacks,
                )
                text = results.generations[0][0].text
            retry += 1

        return {}

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

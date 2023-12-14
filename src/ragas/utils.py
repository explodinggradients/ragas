from __future__ import annotations

import json
import os
import typing as t
import warnings
from dataclasses import dataclass
from functools import lru_cache

from langchain.callbacks.manager import CallbackManager, trace_as_chain_group
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

if t.TYPE_CHECKING:
    from ragas.llms import RagasLLM

DEBUG_ENV_VAR = "RAGAS_DEBUG"
# constant to tell us that there is no key passed to the llm/embeddings
NO_KEY = "no-key"


@lru_cache(maxsize=1)
def get_debug_mode() -> bool:
    if os.environ.get(DEBUG_ENV_VAR, str(False)).lower() == "true":
        return True
    else:
        return False


def load_as_json(text):
    """
    validate and return given text as json
    """

    try:
        return json.loads(text)
    except ValueError as e:
        warnings.warn(f"Invalid json: {e}")

    return {}


JSON_PROMPT = HumanMessagePromptTemplate.from_template(
    """

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
            human_prompt = ChatPromptTemplate.from_messages(
                [JSON_PROMPT.format(input=text)]
            )
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

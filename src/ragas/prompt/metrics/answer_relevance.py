"""Answer Relevance prompt for generating questions and detecting noncommittal responses."""

import json


def answer_relevancy_prompt(response: str) -> str:
    """
    Generate the prompt for answer relevance evaluation.

    Args:
        response: The response text to evaluate

    Returns:
        Formatted prompt string for the LLM
    """
    # Use json.dumps() to safely escape the response string
    safe_response = json.dumps(response)

    return f"""Generate a question for the given answer and Identify if answer is noncommittal. Give noncommittal as 1 if the answer is noncommittal and 0 if the answer is committal. A noncommittal answer is one that is evasive, vague, or ambiguous. For example, "I don't know" or "I'm not sure" are noncommittal answers

--------EXAMPLES-----------
Example 1
Input: {{
    "response": "Albert Einstein was born in Germany."
}}
Output: {{
    "question": "Where was Albert Einstein born?",
    "noncommittal": 0
}}

Example 2
Input: {{
    "response": "I don't know about the  groundbreaking feature of the smartphone invented in 2023 as am unaware of information beyond 2022. "
}}
Output: {{
    "question": "What was the groundbreaking feature of the smartphone invented in 2023?",
    "noncommittal": 1
}}
-----------------------------

Now perform the same with the following input
input: {{
    "response": {safe_response}
}}
Output: """

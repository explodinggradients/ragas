"""Summary Score prompts - V1-identical using exact PydanticPrompt.to_string() output."""

import json
import typing as t


def extract_keyphrases_prompt(text: str) -> str:
    """
    V1-identical keyphrase extraction - matches PydanticPrompt.to_string() exactly.

    Args:
        text: The text to extract keyphrases from

    Returns:
        V1-identical prompt string for the LLM
    """
    # Format input exactly like V1's model_dump_json(indent=4, exclude_none=True)
    safe_text = json.dumps(text)

    return f"""Extract keyphrases of type: Person, Organization, Location, Date/Time, Monetary Values, and Percentages.
Please return the output in a JSON format that complies with the following schema as specified in JSON Schema:
{{"properties": {{"keyphrases": {{"items": {{"type": "string"}}, "title": "Keyphrases", "type": "array"}}}}, "required": ["keyphrases"], "title": "ExtractedKeyphrases", "type": "object"}}Do not use single quotes in your response but double quotes,properly escaped with a backslash.

--------EXAMPLES-----------
Example 1
Input: {{
    "text": "Apple Inc. is a technology company based in Cupertino, California. Founded by Steve Jobs in 1976, it reached a market capitalization of $3 trillion in 2023."
}}
Output: {{
    "keyphrases": [
        "Apple Inc.",
        "Cupertino, California",
        "Steve Jobs",
        "1976",
        "$3 trillion",
        "2023"
    ]
}}
-----------------------------

Now perform the same with the following input
input: {{
    "text": {safe_text}
}}
Output: """


def generate_questions_prompt(text: str, keyphrases: t.List[str]) -> str:
    """
    V1-identical question generation - matches PydanticPrompt.to_string() exactly.

    Args:
        text: The text to generate questions about
        keyphrases: The keyphrases extracted from the text

    Returns:
        V1-identical prompt string for the LLM
    """
    # Format inputs exactly like V1's model_dump_json(indent=4, exclude_none=True)
    safe_text = json.dumps(text)
    safe_keyphrases = json.dumps(keyphrases, indent=4).replace("\n", "\n    ")

    return f"""Based on the given text and keyphrases, generate closed-ended questions that can be answered with '1' if the question can be answered using the text, or '0' if it cannot. The questions should ALWAYS result in a '1' based on the given text.
Please return the output in a JSON format that complies with the following schema as specified in JSON Schema:
{{"properties": {{"questions": {{"items": {{"type": "string"}}, "title": "Questions", "type": "array"}}}}, "required": ["questions"], "title": "QuestionsGenerated", "type": "object"}}Do not use single quotes in your response but double quotes,properly escaped with a backslash.

--------EXAMPLES-----------
Example 1
Input: {{
    "text": "Apple Inc. is a technology company based in Cupertino, California. Founded by Steve Jobs in 1976, it reached a market capitalization of $3 trillion in 2023.",
    "keyphrases": [
        "Apple Inc.",
        "Cupertino, California",
        "Steve Jobs",
        "1976",
        "$3 trillion",
        "2023"
    ]
}}
Output: {{
    "questions": [
        "Is Apple Inc. a technology company?",
        "Is Apple Inc. based in Cupertino, California?",
        "Was Apple Inc. founded by Steve Jobs?",
        "Was Apple Inc. founded in 1976?",
        "Did Apple Inc. reach a market capitalization of $3 trillion?",
        "Did Apple Inc. reach a market capitalization of $3 trillion in 2023?"
    ]
}}
-----------------------------

Now perform the same with the following input
input: {{
    "text": {safe_text},
    "keyphrases": {safe_keyphrases}
}}
Output: """


def generate_answers_prompt(summary: str, questions: t.List[str]) -> str:
    """
    V1-identical answer generation - matches PydanticPrompt.to_string() exactly.

    Args:
        summary: The summary to evaluate
        questions: The questions to check against the summary

    Returns:
        V1-identical prompt string for the LLM
    """
    # Format inputs exactly like V1's model_dump_json(indent=4, exclude_none=True)
    safe_summary = json.dumps(summary)
    safe_questions = json.dumps(questions, indent=4).replace("\n", "\n    ")

    return f"""Based on the list of close-ended '1' or '0' questions, generate a JSON with key 'answers', which is a list of strings that determines whether the provided summary contains sufficient information to answer EACH question. Answers should STRICTLY be either '1' or '0'. Answer '0' if the provided summary does not contain enough information to answer the question and answer '1' if the provided summary can answer the question.
Please return the output in a JSON format that complies with the following schema as specified in JSON Schema:
{{"properties": {{"answers": {{"items": {{"type": "string"}}, "title": "Answers", "type": "array"}}}}, "required": ["answers"], "title": "AnswersGenerated", "type": "object"}}Do not use single quotes in your response but double quotes,properly escaped with a backslash.

--------EXAMPLES-----------
Example 1
Input: {{
    "summary": "Apple Inc. is a technology company based in Cupertino, California. Founded by Steve Jobs in 1976, it reached a market capitalization of $3 trillion in 2023.",
    "questions": [
        "Is Apple Inc. a technology company?",
        "Is Apple Inc. based in Cupertino, California?",
        "Was Apple Inc. founded by Steve Jobs?",
        "Was Apple Inc. founded in 1976?",
        "Did Apple Inc. reach a market capitalization of $3 trillion?",
        "Did Apple Inc. reach a market capitalization of $3 trillion in 2023?",
        "Is Apple Inc. a major software company?",
        "Is Apple Inc. known for the iPhone?",
        "Was Steve Jobs the co-founder of Apple Inc.?"
    ]
}}
Output: {{
    "answers": [
        "1",
        "1",
        "1",
        "1",
        "1",
        "1",
        "0",
        "0",
        "1"
    ]
}}
-----------------------------

Now perform the same with the following input
input: {{
    "summary": {safe_summary},
    "questions": {safe_questions}
}}
Output: """

"""Answer Correctness prompts for classification.

Note: statement_generator_prompt has been moved to ragas.prompt.metrics.common
"""

import json
import typing as t


def correctness_classifier_prompt(
    question: str, answer_statements: t.List[str], ground_truth_statements: t.List[str]
) -> str:
    """
    V1-compatible correctness classifier using exact PydanticPrompt structure.

    Args:
        question: The original question
        answer_statements: List of statements from the answer to evaluate
        ground_truth_statements: List of ground truth reference statements

    Returns:
        V1-identical prompt string for the LLM
    """
    # Format inputs exactly like V1's model_dump_json(indent=4, exclude_none=True)
    safe_question = json.dumps(question)
    safe_answer_statements = json.dumps(answer_statements, indent=4).replace(
        "\n", "\n    "
    )
    safe_ground_truth = json.dumps(ground_truth_statements, indent=4).replace(
        "\n", "\n    "
    )

    return f"""Given a ground truth and an answer, analyze each statement in the answer and classify them in one of the following categories:
    - TP (true positive): statements that are present in answer and present in the ground truth
    - FP (false positive): statements that are present in answer but not present in the ground truth
    - FN (false negative): statements that are present in ground truth but not present in the answer

Please return the output in a JSON format that complies with the following schema as specified in JSON Schema:
{{"$defs": {{"StatementsWithReason": {{"properties": {{"reason": {{"description": "The reason for the verdict", "title": "Reason", "type": "string"}}, "statements": {{"description": "The statement", "items": {{"type": "string"}}, "title": "Statements", "type": "array"}}}}, "required": ["statements", "reason"], "title": "StatementsWithReason", "type": "object"}}}}, "properties": {{"TP": {{"description": "List of true positive statements", "items": {{"$ref": "#/$defs/StatementsWithReason"}}, "title": "Tp", "type": "array"}}, "FP": {{"description": "List of false positive statements", "items": {{"$ref": "#/$defs/StatementsWithReason"}}, "title": "Fp", "type": "array"}}, "FN": {{"description": "List of false negative statements", "items": {{"$ref": "#/$defs/StatementsWithReason"}}, "title": "Fn", "type": "array"}}}}, "required": ["TP", "FP", "FN"], "title": "ClassificationWithReason", "type": "object"}}Do not use single quotes in your response but double quotes,properly escaped with a backslash.

--------EXAMPLES-----------
Example 1
Input: {{
    "question": "What powers the sun and what is its primary function?",
    "answer": [
        "The sun is powered by nuclear fission.",
        "The sun's primary function is to provide light to the solar system."
    ],
    "ground_truth": [
        "The sun is powered by nuclear fusion.",
        "The sun's primary function is to provide light and heat to the solar system."
    ]
}}
Output: {{
    "TP": [
        {{
            "statements": ["The sun's primary function is to provide light to the solar system."],
            "reason": "This statement is present in both the answer and the ground truth. While the ground truth mentions both light and heat, the answer correctly identifies light as a primary function."
        }}
    ],
    "FP": [
        {{
            "statements": ["The sun is powered by nuclear fission."],
            "reason": "This statement is present in the answer but contradicts the ground truth, which states that the sun is powered by nuclear fusion, not fission."
        }}
    ],
    "FN": [
        {{
            "statements": ["The sun's primary function is to provide heat to the solar system."],
            "reason": "The ground truth mentions that the sun provides heat to the solar system, but this information is missing from the answer."
        }}
    ]
}}
-----------------------------

Now perform the same with the following input
input: {{
    "question": {safe_question},
    "answer": {safe_answer_statements},
    "ground_truth": {safe_ground_truth}
}}
Output: """


__all__ = ["correctness_classifier_prompt"]

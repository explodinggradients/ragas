"""Answer Correctness prompts for statement generation and classification."""

import json
import typing as t


def statement_generator_prompt(question: str, answer: str) -> str:
    """
    V1-identical statement generator - matches PydanticPrompt.to_string() exactly.

    Args:
        question: The question being answered
        answer: The answer text to break down into statements

    Returns:
        V1-identical prompt string for the LLM
    """
    # Format inputs exactly like V1's model_dump_json(indent=4, exclude_none=True)
    safe_question = json.dumps(question)
    safe_answer = json.dumps(answer)

    return f"""Given a question and an answer, analyze the complexity of each sentence in the answer. Break down each sentence into one or more fully understandable statements. Ensure that no pronouns are used in any statement. IMPORTANT: Extract statements EXACTLY as they appear in the answer - do not correct factual errors or change any content. Format the outputs in JSON.
Please return the output in a JSON format that complies with the following schema as specified in JSON Schema:
{{"properties": {{"statements": {{"description": "The generated statements", "items": {{"type": "string"}}, "title": "Statements", "type": "array"}}}}, "required": ["statements"], "title": "StatementGeneratorOutput", "type": "object"}}Do not use single quotes in your response but double quotes,properly escaped with a backslash.

--------EXAMPLES-----------
Example 1
Input: {{
    "question": "Who was Albert Einstein and what is he best known for?",
    "answer": "He was a German-born theoretical physicist, widely acknowledged to be one of the greatest and most influential physicists of all time. He was best known for developing the theory of relativity, he also made important contributions to the development of the theory of quantum mechanics."
}}
Output: {{
    "statements": [
        "Albert Einstein was a German-born theoretical physicist.",
        "Albert Einstein is recognized as one of the greatest and most influential physicists of all time.",
        "Albert Einstein was best known for developing the theory of relativity.",
        "Albert Einstein made important contributions to the development of the theory of quantum mechanics."
    ]
}}
-----------------------------

Now perform the same with the following input
input: {{
    "question": {safe_question},
    "answer": {safe_answer}
}}
Output: """


def correctness_classifier_prompt(
    question: str, answer_statements: t.List[str], ground_truth_statements: t.List[str]
) -> str:
    """
    V1-compatible correctness classifier using exact PydanticPrompt structure.

    Args:
        question: The original question
        answer_statements: List of statements from the answer
        ground_truth_statements: List of statements from the ground truth

    Returns:
        V1-compatible prompt string for the LLM
    """
    # Format inputs exactly like V1's model_dump_json(indent=4, exclude_none=True)
    safe_question = json.dumps(question)
    # Format lists with proper indentation to match V1's Pydantic formatting
    safe_answer = json.dumps(answer_statements, indent=4).replace("\n", "\n    ")
    safe_ground_truth = json.dumps(ground_truth_statements, indent=4).replace(
        "\n", "\n    "
    )

    return f"""Given a ground truth and an answer statements, analyze each statement and classify them in one of the following categories: TP (true positive): statements that are present in answer that are also directly supported by the one or more statements in ground truth, FP (false positive): statements present in the answer but not directly supported by any statement in ground truth, FN (false negative): statements found in the ground truth but not present in answer. Each statement can only belong to one of the categories. Provide a reason for each classification.
Please return the output in a JSON format that complies with the following schema as specified in JSON Schema:
{{"$defs": {{"StatementsWithReason": {{"properties": {{"statement": {{"title": "Statement", "type": "string"}}, "reason": {{"title": "Reason", "type": "string"}}}}, "required": ["statement", "reason"], "title": "StatementsWithReason", "type": "object"}}}}, "properties": {{"TP": {{"items": {{"$ref": "#/$defs/StatementsWithReason"}}, "title": "Tp", "type": "array"}}, "FP": {{"items": {{"$ref": "#/$defs/StatementsWithReason"}}, "title": "Fp", "type": "array"}}, "FN": {{"items": {{"$ref": "#/$defs/StatementsWithReason"}}, "title": "Fn", "type": "array"}}}}, "required": ["TP", "FP", "FN"], "title": "ClassificationWithReason", "type": "object"}}Do not use single quotes in your response but double quotes,properly escaped with a backslash.

--------EXAMPLES-----------
Example 1
Input: {{
    "question": "What powers the sun and what is its primary function?",
    "answer": [
        "The sun is powered by nuclear fission, similar to nuclear reactors on Earth.",
        "The primary function of the sun is to provide light to the solar system."
    ],
    "ground_truth": [
        "The sun is powered by nuclear fusion, where hydrogen atoms fuse to form helium.",
        "This fusion process in the sun's core releases a tremendous amount of energy.",
        "The energy from the sun provides heat and light, which are essential for life on Earth.",
        "The sun's light plays a critical role in Earth's climate system.",
        "Sunlight helps to drive the weather and ocean currents."
    ]
}}
Output: {{
    "TP": [
        {{
            "statement": "The primary function of the sun is to provide light to the solar system.",
            "reason": "This statement is somewhat supported by the ground truth mentioning the sun providing light and its roles, though it focuses more broadly on the sun's energy."
        }}
    ],
    "FP": [
        {{
            "statement": "The sun is powered by nuclear fission, similar to nuclear reactors on Earth.",
            "reason": "This statement is incorrect and contradicts the ground truth which states that the sun is powered by nuclear fusion."
        }}
    ],
    "FN": [
        {{
            "statement": "The sun is powered by nuclear fusion, where hydrogen atoms fuse to form helium.",
            "reason": "This accurate description of the sun's power source is not included in the answer."
        }},
        {{
            "statement": "This fusion process in the sun's core releases a tremendous amount of energy.",
            "reason": "This process and its significance are not mentioned in the answer."
        }},
        {{
            "statement": "The energy from the sun provides heat and light, which are essential for life on Earth.",
            "reason": "The answer only mentions light, omitting the essential aspects of heat and its necessity for life, which the ground truth covers."
        }},
        {{
            "statement": "The sun's light plays a critical role in Earth's climate system.",
            "reason": "This broader impact of the sun's light on Earth's climate system is not addressed in the answer."
        }},
        {{
            "statement": "Sunlight helps to drive the weather and ocean currents.",
            "reason": "The effect of sunlight on weather patterns and ocean currents is omitted in the answer."
        }}
    ]
}}

Example 2
Input: {{
    "question": "What is the boiling point of water?",
    "answer": [
        "The boiling point of water is 100 degrees Celsius at sea level"
    ],
    "ground_truth": [
        "The boiling point of water is 100 degrees Celsius (212 degrees Fahrenheit) at sea level.",
        "The boiling point of water can change with altitude."
    ]
}}
Output: {{
    "TP": [
        {{
            "statement": "The boiling point of water is 100 degrees Celsius at sea level",
            "reason": "This statement is directly supported by the ground truth which specifies the boiling point of water as 100 degrees Celsius at sea level."
        }}
    ],
    "FP": [],
    "FN": [
        {{
            "statement": "The boiling point of water can change with altitude.",
            "reason": "This additional information about how the boiling point of water can vary with altitude is not mentioned in the answer."
        }}
    ]
}}
-----------------------------

Now perform the same with the following input
input: {{
    "question": {safe_question},
    "answer": {safe_answer},
    "ground_truth": {safe_ground_truth}
}}
Output: """

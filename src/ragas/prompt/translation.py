"""
Translation utilities for SimplePydanticPrompt.
"""

import typing as t

from ragas.llms.base import InstructorBaseRagasLLM


async def translate_prompt_content(
    strings: t.List[str], target_language: str, llm: InstructorBaseRagasLLM
) -> t.List[str]:
    """
    Translate a list of strings using the provided LLM.

    Args:
        strings: List of strings to translate
        target_language: Target language for translation
        llm: LLM to use for translation

    Returns:
        List of translated strings in the same order
    """
    if not strings:
        return []

    # Import here to avoid circular imports
    from .simple_pydantic_prompt import (
        ToTranslate,
        Translated,
        translate_statements_prompt,
    )

    # Use the translation prompt with InstructorBaseRagasLLM
    translation_input = ToTranslate(target_language=target_language, statements=strings)

    # Generate translation using structured output
    result = await llm.agenerate(
        translate_statements_prompt.to_string(translation_input), Translated
    )
    return result.statements

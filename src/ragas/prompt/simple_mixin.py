"""
Simplified PromptMixin that works with SimplePydanticPrompt.
Focuses on core functionality without bloat.
"""

from __future__ import annotations

import inspect
import logging
import typing as t

from .simple_pydantic_prompt import SimplePydanticPrompt

if t.TYPE_CHECKING:
    from ragas.llms.base import InstructorBaseRagasLLM

logger = logging.getLogger(__name__)


class SimplePromptMixin:
    """
    Simplified mixin class for classes that have prompts.

    Provides essential prompt management functionality:
    - Get prompts from class attributes
    - Set/modify prompts
    - Translate prompts to different languages

    Works with SimplePydanticPrompt instances.
    """

    def get_prompts(self) -> t.Dict[str, SimplePydanticPrompt]:
        """
        Get all prompts from this class.

        Returns:
            Dictionary mapping prompt names to prompt instances
        """
        prompts = {}

        for attr_name, attr_value in inspect.getmembers(self):
            if isinstance(attr_value, SimplePydanticPrompt):
                # Use the prompt's name if it has one, otherwise use attribute name
                prompt_name = attr_value.name or attr_name
                prompts[prompt_name] = attr_value

        return prompts

    def set_prompts(self, **prompts: SimplePydanticPrompt) -> None:
        """
        Set/update prompts on this class.

        Args:
            **prompts: Keyword arguments where keys are prompt names and
                      values are SimplePydanticPrompt instances

        Raises:
            ValueError: If prompt name doesn't exist or value is not a SimplePydanticPrompt
        """
        available_prompts = self.get_prompts()
        name_to_attr = self._get_prompt_name_to_attr_mapping()

        for prompt_name, new_prompt in prompts.items():
            if prompt_name not in available_prompts:
                available_names = list(available_prompts.keys())
                raise ValueError(
                    f"Prompt '{prompt_name}' not found. Available prompts: {available_names}"
                )

            if not isinstance(new_prompt, SimplePydanticPrompt):
                raise ValueError(
                    f"Prompt '{prompt_name}' must be a SimplePydanticPrompt instance"
                )

            # Set the prompt on the class
            attr_name = name_to_attr[prompt_name]
            setattr(self, attr_name, new_prompt)

    async def adapt_prompts(
        self,
        target_language: str,
        llm: InstructorBaseRagasLLM,
        adapt_instruction: bool = False,
    ) -> t.Dict[str, SimplePydanticPrompt]:
        """
        Translate all prompts to the target language.

        Args:
            target_language: Target language for translation
            llm: LLM to use for translation
            adapt_instruction: Whether to translate instructions as well as examples

        Returns:
            Dictionary of translated prompts
        """
        prompts = self.get_prompts()
        adapted_prompts = {}

        for prompt_name, prompt in prompts.items():
            try:
                adapted_prompt = await prompt.adapt(
                    target_language, llm, adapt_instruction
                )
                adapted_prompts[prompt_name] = adapted_prompt
            except Exception as e:
                logger.warning(f"Failed to adapt prompt '{prompt_name}': {e}")
                # Keep original prompt on failure
                adapted_prompts[prompt_name] = prompt

        return adapted_prompts

    def set_adapted_prompts(
        self, adapted_prompts: t.Dict[str, SimplePydanticPrompt]
    ) -> None:
        """
        Set adapted/translated prompts on this class.

        Args:
            adapted_prompts: Dictionary of translated prompts from adapt_prompts()
        """
        self.set_prompts(**adapted_prompts)

    def modify_prompt(
        self,
        prompt_name: str,
        instruction: t.Optional[str] = None,
        examples: t.Optional[t.List] = None,
    ) -> None:
        """
        Modify a specific prompt's instruction or examples.

        Args:
            prompt_name: Name of the prompt to modify
            instruction: New instruction (if provided)
            examples: New examples (if provided)
        """
        current_prompts = self.get_prompts()

        if prompt_name not in current_prompts:
            available_names = list(current_prompts.keys())
            raise ValueError(
                f"Prompt '{prompt_name}' not found. Available prompts: {available_names}"
            )

        current_prompt = current_prompts[prompt_name]
        modified_prompt = current_prompt.copy_with_modifications(
            instruction=instruction, examples=examples
        )

        self.set_prompts(**{prompt_name: modified_prompt})

    def _get_prompt_name_to_attr_mapping(self) -> t.Dict[str, str]:
        """Get mapping from prompt names to attribute names."""
        mapping = {}

        for attr_name, attr_value in inspect.getmembers(self):
            if isinstance(attr_value, SimplePydanticPrompt):
                prompt_name = attr_value.name or attr_name
                mapping[prompt_name] = attr_name

        return mapping

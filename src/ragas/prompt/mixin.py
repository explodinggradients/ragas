from __future__ import annotations

import inspect
import typing as t

from .pydantic_prompt import PydanticPrompt

if t.TYPE_CHECKING:
    from ragas.llms.base import BaseRagasLLM


class PromptMixin:
    def get_prompts(self) -> t.Dict[str, PydanticPrompt]:
        prompts = {}
        for name, value in inspect.getmembers(self):
            if isinstance(value, PydanticPrompt):
                prompts.update({name: value})
        return prompts

    def set_prompts(self, **prompts):
        available_prompts = self.get_prompts()
        for key, value in prompts.items():
            if key not in available_prompts:
                raise ValueError(
                    f"Prompt with name '{key}' does not exist. Use get_prompts() to see available prompts."
                )
            if not isinstance(value, PydanticPrompt):
                raise ValueError(
                    f"Prompt with name '{key}' must be an instance of 'ragas.prompt.PydanticPrompt'"
                )
            setattr(self, key, value)

    async def adapt_prompts(
        self, language: str, llm: BaseRagasLLM
    ) -> t.Dict[str, PydanticPrompt]:
        prompts = self.get_prompts()
        adapted_prompts = {}
        for name, prompt in prompts.items():
            adapted_prompt = await prompt.adapt(language, llm)
            adapted_prompts[name] = adapted_prompt

        return adapted_prompts

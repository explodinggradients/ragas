import inspect
import typing as t

from .base import PydanticPrompt


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

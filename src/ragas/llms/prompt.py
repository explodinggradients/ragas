import typing as t

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompt_values import PromptValue
from pydantic import Field


class Prompt(PromptValue):
    instruction: str
    examples: t.List[t.Dict[str, t.Any]] = Field(default_factory=list, repr=False)
    input_keys: t.List[str] = Field(default_factory=list, repr=False)
    output_keys: t.List[str] = Field(default_factory=list, repr=False)

    def to_string(self) -> str:
        return self.instruction

    def to_messages(self) -> t.List[BaseMessage]:
        return [HumanMessage(content=self.instruction)]

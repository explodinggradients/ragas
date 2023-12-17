import typing as t

from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompt_values import PromptValue
from pydantic import Field


class Prompt(PromptValue):
    chat_prompt_template: ChatPromptTemplate
    instruction: t.Optional[str] = None
    examples: t.List[t.Dict[str, t.Any]] = Field(default_factory=list, repr=False)
    input_keys: t.List[str] = Field(default_factory=list, repr=False)
    output_keys: t.List[str] = Field(default_factory=list, repr=False)

    def to_string(self) -> str:
        return self.chat_prompt_template.format()

    def to_messages(self) -> t.List[BaseMessage]:
        return self.chat_prompt_template.format_messages()

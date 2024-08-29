from typing import Any, Dict, List, Literal, Optional, Union

from langchain_core.pydantic_v1 import BaseModel


class Message(BaseModel):
    content: str
    metadata: Optional[Dict[str, Any]] = None


class ToolCall(BaseModel):
    name: str
    args: Dict[str, Union[str, int, float]]


class HumanMessage(Message):
    type: Literal["human"] = "human"

    def pretty_repr(self):
        return f"Human: {self.content}"


class ToolMessage(Message):
    type: Literal["tool"] = "tool"

    def pretty_repr(self):
        return f"ToolOutput: {self.content}"


class AIMessage(Message):
    type: Literal["ai"] = "ai"
    tool_calls: Optional[List[ToolCall]] = None
    metadata: Optional[Dict[str, Any]] = None

    def dict(self, **kwargs):
        content = (
            self.content
            if self.tool_calls is None
            else {
                "text": self.content,
                "tool_calls": [tc.dict() for tc in self.tool_calls],
            }
        )
        return {"content": content, "type": self.type}

    def pretty_repr(self):
        lines = []
        if self.content != "":
            lines.append(f"AI: {self.content}")
        if self.tool_calls is not None:
            lines.append("Tools:")
            for tc in self.tool_calls:
                lines.append(f"  {tc.name}: {tc.args}")

        return "\n".join(lines)

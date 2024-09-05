import typing as t

from langchain_core.pydantic_v1 import BaseModel


class Message(BaseModel):
    content: str
    metadata: t.Optional[t.Dict[str, t.Any]] = None


class ToolCall(BaseModel):
    name: str
    args: t.Dict[str, t.Union[str, int, float]]


class HumanMessage(Message):
    type: t.Literal["human"] = "human"

    def pretty_repr(self):
        return f"Human: {self.content}"


class ToolMessage(Message):
    type: t.Literal["tool"] = "tool"

    def pretty_repr(self):
        return f"ToolOutput: {self.content}"


class AIMessage(Message):
    type: t.Literal["ai"] = "ai"
    tool_calls: t.Optional[t.List[ToolCall]] = None
    metadata: t.Optional[t.Dict[str, t.Any]] = None

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

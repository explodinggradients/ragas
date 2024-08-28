from typing import Any, Dict, List, Literal, Optional, Union

from datasets import Dataset
from langchain_core.pydantic_v1 import BaseModel, validator


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


class EvaluationSample(BaseModel):
    user_input: Optional[str] = None
    retrieved_contexts: Optional[List[str]] = None
    ground_truth_contexts: Optional[List[str]] = None
    response: Optional[str] = None
    reference: Optional[str] = None
    rubric: Optional[Dict[str, str]] = None

    def dict(self, **kwargs):
        row = {
            "user_input": self.user_input,
            "retrieved_contexts": self.retrieved_contexts,
            "ground_truth_contexts": self.ground_truth_contexts,
            "response": self.response,
            "reference": self.reference,
            "rubric": self.rubric,
        }
        row = {k: v for k, v in row.items() if v is not None}
        return row


class WorkflowEvaluationSample(BaseModel):
    user_input: List[Union[HumanMessage, AIMessage, ToolMessage]]
    reference: Optional[str] = None

    @validator("user_input")
    def validate_messages(cls, messages):
        if not (
            isinstance(m, (HumanMessage, AIMessage, ToolMessage)) for m in messages
        ):
            raise ValueError(
                "All inputs must be instances of HumanMessage, AIMessage, or ToolMessage."
            )

        prev_message = None
        for m in messages:
            if isinstance(m, ToolMessage):
                if not isinstance(prev_message, AIMessage):
                    raise ValueError(
                        "ToolMessage instances must be preceded by an AIMessage instance."
                    )
                if prev_message.tool_calls is None:
                    raise ValueError(
                        f"ToolMessage instances must be preceded by an AIMessage instance with tool_calls. Got {prev_message}"
                    )
            prev_message = m

        return messages

    def to_messages(self):
        return [m.dict() for m in self.user_input]

    def pretty_repr(self):
        lines = []
        for m in self.user_input:
            lines.append(m.pretty_repr())

        return "\n".join(lines)


class EvaluationDataset(BaseModel):
    samples: List[EvaluationSample]

    def to_hf_dataset(self):
        rows = [sample.dict() for sample in self.samples]
        return Dataset.from_list(rows)

    def features(self):
        return self.to_hf_dataset().features.keys()

from typing import Dict, List, Literal, Optional, Union

from datasets import Dataset
from langchain_core.pydantic_v1 import BaseModel, validator


class Message(BaseModel):
    content: str


class ToolCall(BaseModel):
    name: str
    args: Dict[str, Union[str, int, float]]


class HumanMessage(Message):
    type: Literal["human"] = "human"


class ToolMessage(Message):
    type: Literal["tool"] = "tool"


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


class EvaluationSample(BaseModel):
    user_input: Optional[str] = None
    retrieved_contexts: Optional[List[str]] = None
    ground_truth_contexts: Optional[List[str]] = None
    response: Optional[str] = None
    reference: Optional[str] = None
    rubric: Optional[Dict[str, str]] = None

    def to_dict(self):
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

    @validator("user_input")
    def validate_messages(cls, v):
        if not (isinstance(m, (HumanMessage, AIMessage, ToolMessage)) for m in v):
            raise ValueError(
                f"All inputs must be instances of HumanMessage, AIMessage, or ToolMessage. Got: {v}"
            )
        return v

    def to_messages(self):
        return [m.dict() for m in self.user_input]


class EvaluationDataset(BaseModel):
    samples: List[EvaluationSample]

    def to_hf_dataset(self):
        rows = [sample.to_dict() for sample in self.samples]
        return Dataset.from_list(rows)

    def features(self):
        return self.to_hf_dataset().features.keys()

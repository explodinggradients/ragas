import json
from typing import Dict, Iterator, List, Optional, Union

from datasets import Dataset
from langchain_core.pydantic_v1 import BaseModel, validator

from ragas.messages import AIMessage, HumanMessage, ToolMessage


class BaseEvalSample(BaseModel):
    ...


class SingleTurnSample(BaseEvalSample):
    user_input: Optional[str] = None
    retrieved_contexts: Optional[List[str]] = None
    reference_contexts: Optional[List[str]] = None
    response: Optional[str] = None
    multi_responses: Optional[List[str]] = None
    reference: Optional[str] = None
    rubric: Optional[Dict[str, str]] = None

    def dict(self, **kwargs):
        row = super().dict(**kwargs)
        row = {k: v for k, v in row.items() if v is not None}
        return row

    def features(self):
        return self.dict().keys()


class MultiTurnSample(BaseEvalSample):
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
    samples: List[BaseEvalSample]

    def __post_init__(self):
        first_sample_type = type(self.samples[0])
        if not all(isinstance(sample, first_sample_type) for sample in self.samples):
            raise ValueError("All samples must be of the same type")

    def get_sample_type(self):
        return type(self.samples[0])

    def to_hf_dataset(self):
        rows = [sample.dict() for sample in self.samples]

        if self.get_sample_type() == MultiTurnSample:
            for sample in rows:
                for item in sample["user_input"]:
                    if not isinstance(item["content"], str):
                        item["content"] = json.dumps(item["content"])

        return Dataset.from_list(rows)

    def features(self):
        return self.samples[0].dict().keys()

    @classmethod
    def from_list(cls, mapping: List[Dict]):
        samples = []
        if all(
            "user_input" in item and isinstance(mapping[0]["user_input"], list)
            for item in mapping
        ):
            samples.extend(MultiTurnSample(**sample) for sample in mapping)
        else:
            samples.extend(SingleTurnSample(**sample) for sample in mapping)
        return cls(samples=samples)

    def __iter__(self) -> Iterator[BaseEvalSample]:
        return iter(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> BaseEvalSample:
        return self.samples[idx]

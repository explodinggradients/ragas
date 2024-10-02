from __future__ import annotations

import json
import typing as t

from pydantic import BaseModel, field_validator

from ragas.messages import AIMessage, HumanMessage, ToolCall, ToolMessage

if t.TYPE_CHECKING:
    from datasets import Dataset as HFDataset
    from pandas import DataFrame as PandasDataframe


class BaseEvalSample(BaseModel):
    def to_dict(self) -> t.Dict:
        """
        Get the dictionary representation of the sample without attributes that are None.
        """
        return self.model_dump(exclude_none=True)

    def get_features(self) -> t.List[str]:
        """
        Get the features of the sample that are not None.
        """
        return list(self.to_dict().keys())


class SingleTurnSample(BaseEvalSample):
    """
    Represents evaluation samples for single-turn interactions.
    """

    user_input: t.Optional[str] = None
    retrieved_contexts: t.Optional[t.List[str]] = None
    reference_contexts: t.Optional[t.List[str]] = None
    response: t.Optional[str] = None
    multi_responses: t.Optional[t.List[str]] = None
    reference: t.Optional[str] = None
    rubric: t.Optional[t.Dict[str, str]] = None


class MultiTurnSample(BaseEvalSample):
    user_input: t.List[t.Union[HumanMessage, AIMessage, ToolMessage]]
    reference: t.Optional[str] = None
    reference_tool_calls: t.Optional[t.List[ToolCall]] = None
    rubrics: t.Optional[t.Dict[str, str]] = None
    reference_topics: t.Optional[t.List[str]] = None

    @field_validator("user_input")
    @classmethod
    def validate_user_input(
        cls,
        messages: t.List[t.Union[HumanMessage, AIMessage, ToolMessage]],
    ) -> t.List[t.Union[HumanMessage, AIMessage, ToolMessage]]:
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
        return [m.model_dump() for m in self.user_input]

    def pretty_repr(self):
        lines = []
        for m in self.user_input:
            lines.append(m.pretty_repr())

        return "\n".join(lines)


class EvaluationDataset(BaseModel):
    samples: t.List[BaseEvalSample]

    @field_validator("samples")
    def validate_samples(
        cls, samples: t.List[BaseEvalSample]
    ) -> t.List[BaseEvalSample]:
        if len(samples) == 0:
            return samples

        first_sample_type = type(samples[0])
        if not all(isinstance(sample, first_sample_type) for sample in samples):
            raise ValueError("All samples must be of the same type")

        return samples

    def get_sample_type(self):
        return type(self.samples[0])

    def _to_list(self) -> t.List[t.Dict]:
        rows = [sample.model_dump() for sample in self.samples]

        if self.get_sample_type() == MultiTurnSample:
            for sample in rows:
                for item in sample["user_input"]:
                    if not isinstance(item["content"], str):
                        item["content"] = json.dumps(item["content"])

        return rows

    def to_hf_dataset(self) -> HFDataset:
        try:
            from datasets import Dataset as HFDataset
        except ImportError:
            raise ImportError(
                "datasets is not installed. Please install it to use this function."
            )

        return HFDataset.from_list(self._to_list())

    def to_pandas(self) -> PandasDataframe:
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is not installed. Please install it to use this function."
            )

        data = self._to_list()
        return pd.DataFrame(data)

    def features(self):
        return self.samples[0].get_features()

    @classmethod
    def from_list(cls, mapping: t.List[t.Dict]):
        samples = []
        if all(
            "user_input" in item and isinstance(mapping[0]["user_input"], list)
            for item in mapping
        ):
            samples.extend(MultiTurnSample(**sample) for sample in mapping)
        else:
            samples.extend(SingleTurnSample(**sample) for sample in mapping)
        return cls(samples=samples)

    @classmethod
    def from_dict(cls, mapping: t.Dict):
        samples = []
        if all(
            "user_input" in item and isinstance(mapping[0]["user_input"], list)
            for item in mapping
        ):
            samples.extend(MultiTurnSample(**sample) for sample in mapping)
        else:
            samples.extend(SingleTurnSample(**sample) for sample in mapping)
        return cls(samples=samples)

    def __iter__(self) -> t.Iterator[BaseEvalSample]:  # type: ignore
        return iter(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> BaseEvalSample:
        return self.samples[idx]

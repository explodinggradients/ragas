from __future__ import annotations

import json
import typing as t
from dataclasses import dataclass, field

from datasets import Dataset as HFDataset
from datasets import concatenate_datasets
from pydantic import BaseModel, field_validator

from ragas.messages import AIMessage, HumanMessage, ToolCall, ToolMessage
from ragas.utils import safe_nanmean

if t.TYPE_CHECKING:
    from datasets import Dataset as HFDataset
    from pandas import DataFrame as PandasDataframe

    from ragas.cost import CostCallbackHandler, TokenUsage


class BaseEvalSample(BaseModel):
    """
    Base class for evaluation samples.
    """

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

    Attributes
    ----------
    user_input : Optional[str]
        The input query from the user.
    retrieved_contexts : Optional[List[str]]
        List of contexts retrieved for the query.
    reference_contexts : Optional[List[str]]
        List of reference contexts for the query.
    response : Optional[str]
        The generated response for the query.
    multi_responses : Optional[List[str]]
        List of multiple responses generated for the query.
    reference : Optional[str]
        The reference answer for the query.
    rubric : Optional[Dict[str, str]]
        Evaluation rubric for the sample.
    """

    user_input: t.Optional[str] = None
    retrieved_contexts: t.Optional[t.List[str]] = None
    reference_contexts: t.Optional[t.List[str]] = None
    response: t.Optional[str] = None
    multi_responses: t.Optional[t.List[str]] = None
    reference: t.Optional[str] = None
    rubric: t.Optional[t.Dict[str, str]] = None


class MultiTurnSample(BaseEvalSample):
    """
    Represents evaluation samples for multi-turn interactions.

    Attributes
    ----------
    user_input : List[Union[HumanMessage, AIMessage, ToolMessage]]
        A list of messages representing the conversation turns.
    reference : Optional[str], optional
        The reference answer or expected outcome for the conversation.
    reference_tool_calls : Optional[List[ToolCall]], optional
        A list of expected tool calls for the conversation.
    rubrics : Optional[Dict[str, str]], optional
        Evaluation rubrics for the conversation.
    reference_topics : Optional[List[str]], optional
        A list of reference topics for the conversation.
    """

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
        """Validates the user input messages."""
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
        """Converts the user input messages to a list of dictionaries."""
        return [m.model_dump() for m in self.user_input]

    def pretty_repr(self):
        """Returns a pretty string representation of the conversation."""
        lines = []
        for m in self.user_input:
            lines.append(m.pretty_repr())

        return "\n".join(lines)


class EvaluationDataset(BaseModel):
    """
    Represents a dataset of evaluation samples.

    Parameters
    ----------
    samples : List[BaseEvalSample]
        A list of evaluation samples.

    Attributes
    ----------
    samples : List[BaseEvalSample]
        A list of evaluation samples.

    Methods
    -------
    validate_samples(samples)
        Validates that all samples are of the same type.
    get_sample_type()
        Returns the type of the samples in the dataset.
    to_hf_dataset()
        Converts the dataset to a Hugging Face Dataset.
    to_pandas()
        Converts the dataset to a pandas DataFrame.
    features()
        Returns the features of the samples.
    from_list(mapping)
        Creates an EvaluationDataset from a list of dictionaries.
    from_dict(mapping)
        Creates an EvaluationDataset from a dictionary.
    """

    samples: t.List[BaseEvalSample]

    @field_validator("samples")
    def validate_samples(
        cls, samples: t.List[BaseEvalSample]
    ) -> t.List[BaseEvalSample]:
        """Validates that all samples are of the same type."""
        if len(samples) == 0:
            return samples

        first_sample_type = type(samples[0])
        if not all(isinstance(sample, first_sample_type) for sample in samples):
            raise ValueError("All samples must be of the same type")

        return samples

    def get_sample_type(self):
        """Returns the type of the samples in the dataset."""
        return type(self.samples[0])

    def _to_list(self) -> t.List[t.Dict]:
        """Converts the samples to a list of dictionaries."""
        rows = [sample.to_dict() for sample in self.samples]

        if self.get_sample_type() == MultiTurnSample:
            for sample in rows:
                for item in sample["user_input"]:
                    if not isinstance(item["content"], str):
                        item["content"] = json.dumps(item["content"])

        return rows

    def to_hf_dataset(self) -> HFDataset:
        """Converts the dataset to a Hugging Face Dataset."""
        try:
            from datasets import Dataset as HFDataset
        except ImportError:
            raise ImportError(
                "datasets is not installed. Please install it to use this function."
            )

        return HFDataset.from_list(self._to_list())

    def to_pandas(self) -> PandasDataframe:
        """Converts the dataset to a pandas DataFrame."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is not installed. Please install it to use this function."
            )

        data = self._to_list()
        return pd.DataFrame(data)

    def features(self):
        """Returns the features of the samples."""
        return self.samples[0].get_features()

    @classmethod
    def from_list(cls, mapping: t.List[t.Dict]):
        """Creates an EvaluationDataset from a list of dictionaries."""
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
        """Creates an EvaluationDataset from a dictionary."""
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


class EvaluationResult(BaseModel):
    """
    A class to store and process the results of the evaluation.

    Attributes
    ----------
    scores : Dataset
        The dataset containing the scores of the evaluation.
    dataset : Dataset, optional
        The original dataset used for the evaluation. Default is None.
    binary_columns : list of str, optional
        List of columns that are binary metrics. Default is an empty list.
    cost_cb : CostCallbackHandler, optional
        The callback handler for cost computation. Default is None.
    """

    scores: HFDataset
    dataset: t.Optional[HFDataset] = None
    binary_columns: t.List[str] = field(default_factory=list)
    cost_cb: t.Optional[CostCallbackHandler] = None

    # def __post_init__(self):
    #     values = []
    #     for cn in self.scores[0].keys():
    #         value = safe_nanmean(self.scores[cn])
    #         self[cn] = value
    #         if cn not in self.binary_columns:
    #             value = t.cast(float, value)
    #             values.append(value + 1e-10)

    def to_pandas(self, batch_size: int | None = None, batched: bool = False):
        """
        Convert the result to a pandas DataFrame.

        Parameters
        ----------
        batch_size : int, optional
            The batch size for conversion. Default is None.
        batched : bool, optional
            Whether to convert in batches. Default is False.

        Returns
        -------
        pandas.DataFrame
            The result as a pandas DataFrame.

        Raises
        ------
        ValueError
            If the dataset is not provided.
        """
        if self.dataset is None:
            raise ValueError("dataset is not provided for the results class")
        assert self.scores.shape[0] == self.dataset.shape[0]
        result_ds = concatenate_datasets([self.dataset, self.scores], axis=1)

        return result_ds.to_pandas(batch_size=batch_size, batched=batched)

    def total_tokens(self) -> t.Union[t.List[TokenUsage], TokenUsage]:
        """
        Compute the total tokens used in the evaluation.

        Returns
        -------
        list of TokenUsage or TokenUsage
            The total tokens used.

        Raises
        ------
        ValueError
            If the cost callback handler is not provided.
        """
        if self.cost_cb is None:
            raise ValueError(
                "The evaluate() run was not configured for computing cost. Please provide a token_usage_parser function to evaluate() to compute cost."
            )
        return self.cost_cb.total_tokens()

    def total_cost(
        self,
        cost_per_input_token: t.Optional[float] = None,
        cost_per_output_token: t.Optional[float] = None,
        per_model_costs: t.Dict[str, t.Tuple[float, float]] = {},
    ) -> float:
        """
        Compute the total cost of the evaluation.

        Parameters
        ----------
        cost_per_input_token : float, optional
            The cost per input token. Default is None.
        cost_per_output_token : float, optional
            The cost per output token. Default is None.
        per_model_costs : dict of str to tuple of float, optional
            The per model costs. Default is an empty dictionary.

        Returns
        -------
        float
            The total cost of the evaluation.

        Raises
        ------
        ValueError
            If the cost callback handler is not provided.
        """
        if self.cost_cb is None:
            raise ValueError(
                "The evaluate() run was not configured for computing cost. Please provide a token_usage_parser function to evaluate() to compute cost."
            )
        return self.cost_cb.total_cost(
            cost_per_input_token, cost_per_output_token, per_model_costs
        )

    def __repr__(self) -> str:
        scores = self.model_dump()
        score_strs = [f"'{k}': {v:0.4f}" for k, v in scores.items()]
        return "{" + ", ".join(score_strs) + "}"

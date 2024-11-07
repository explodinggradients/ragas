from __future__ import annotations

import json
import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from datasets import Dataset as HFDataset
from pydantic import BaseModel, field_validator

from ragas.callbacks import ChainRunEncoder, parse_run_traces
from ragas.cost import CostCallbackHandler
from ragas.messages import AIMessage, HumanMessage, ToolCall, ToolMessage
from ragas.utils import RAGAS_API_URL, safe_nanmean

if t.TYPE_CHECKING:
    from pathlib import Path

    from datasets import Dataset as HFDataset
    from pandas import DataFrame as PandasDataframe

    from ragas.callbacks import ChainRun
    from ragas.cost import TokenUsage


class BaseSample(BaseModel):
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


class SingleTurnSample(BaseSample):
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


class MultiTurnSample(BaseSample):
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


Sample = t.TypeVar("Sample", bound=BaseSample)
T = t.TypeVar("T", bound="RagasDataset")


@dataclass
class RagasDataset(ABC, t.Generic[Sample]):
    samples: t.List[Sample]

    def __post_init__(self):
        self.samples = self.validate_samples(self.samples)

    @abstractmethod
    def to_list(self) -> t.List[t.Dict]:
        """Converts the samples to a list of dictionaries."""
        pass

    @classmethod
    @abstractmethod
    def from_list(cls: t.Type[T], data: t.List[t.Dict]) -> T:
        """Creates an RagasDataset from a list of dictionaries."""
        pass

    def validate_samples(self, samples: t.List[Sample]) -> t.List[Sample]:
        """Validates that all samples are of the same type."""
        if len(samples) == 0:
            return samples

        first_sample_type = type(self.samples[0])
        if not all(isinstance(sample, first_sample_type) for sample in self.samples):
            raise ValueError("All samples must be of the same type")

        return samples

    def get_sample_type(self) -> t.Type[Sample]:
        """Returns the type of the samples in the dataset."""
        return type(self.samples[0])

    def to_hf_dataset(self) -> HFDataset:
        """Converts the dataset to a Hugging Face Dataset."""
        try:
            from datasets import Dataset as HFDataset
        except ImportError:
            raise ImportError(
                "datasets is not installed. Please install it to use this function."
            )

        return HFDataset.from_list(self.to_list())

    @classmethod
    def from_hf_dataset(cls: t.Type[T], dataset: HFDataset) -> T:
        """Creates an EvaluationDataset from a Hugging Face Dataset."""
        return cls.from_list(dataset.to_list())

    def to_pandas(self) -> PandasDataframe:
        """Converts the dataset to a pandas DataFrame."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is not installed. Please install it to use this function."
            )

        data = self.to_list()
        return pd.DataFrame(data)

    @classmethod
    def from_pandas(cls, dataframe: PandasDataframe):
        """Creates an EvaluationDataset from a pandas DataFrame."""
        return cls.from_list(dataframe.to_dict(orient="records"))

    def features(self):
        """Returns the features of the samples."""
        return self.samples[0].get_features()

    @classmethod
    def from_dict(cls: t.Type[T], mapping: t.Dict) -> T:
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

    def to_csv(self, path: t.Union[str, Path]):
        """Converts the dataset to a CSV file."""
        import csv

        data = self.to_list()
        if not data:
            return

        fieldnames = data[0].keys()

        with open(path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in data:
                writer.writerow(row)

    def to_jsonl(self, path: t.Union[str, Path]):
        """Converts the dataset to a JSONL file."""
        with open(path, "w") as jsonlfile:
            for sample in self.to_list():
                jsonlfile.write(json.dumps(sample, ensure_ascii=False) + "\n")

    @classmethod
    def from_jsonl(cls: t.Type[T], path: t.Union[str, Path]) -> T:
        """Creates an EvaluationDataset from a JSONL file."""
        with open(path, "r") as jsonlfile:
            data = [json.loads(line) for line in jsonlfile]
        return cls.from_list(data)

    def __iter__(self) -> t.Iterator[Sample]:  # type: ignore
        return iter(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __str__(self) -> str:
        return f"EvaluationDataset(features={self.features()}, len={len(self.samples)})"

    def __repr__(self) -> str:
        return self.__str__()


SingleTurnSampleOrMultiTurnSample = t.Union[SingleTurnSample, MultiTurnSample]


@dataclass
class EvaluationDataset(RagasDataset[SingleTurnSampleOrMultiTurnSample]):
    """
    Represents a dataset of evaluation samples.

    Attributes
    ----------
    samples : List[BaseSample]
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
    to_csv(path)
        Converts the dataset to a CSV file.
    to_jsonl(path)
        Converts the dataset to a JSONL file.
    from_jsonl(path)
        Creates an EvaluationDataset from a JSONL file.
    """

    @t.overload
    def __getitem__(self, idx: int) -> SingleTurnSampleOrMultiTurnSample: ...

    @t.overload
    def __getitem__(self, idx: slice) -> "EvaluationDataset": ...

    def __getitem__(
        self, idx: t.Union[int, slice]
    ) -> t.Union[SingleTurnSampleOrMultiTurnSample, "EvaluationDataset"]:
        if isinstance(idx, int):
            return self.samples[idx]
        elif isinstance(idx, slice):
            return type(self)(samples=self.samples[idx])
        else:
            raise TypeError("Index must be int or slice")

    def is_multi_turn(self) -> bool:
        return self.get_sample_type() == MultiTurnSample

    def to_list(self) -> t.List[t.Dict]:
        rows = [sample.to_dict() for sample in self.samples]

        if self.get_sample_type() == MultiTurnSample:
            for sample in rows:
                for item in sample["user_input"]:
                    if not isinstance(item["content"], str):
                        item["content"] = json.dumps(
                            item["content"], ensure_ascii=False
                        )

        return rows

    @classmethod
    def from_list(cls, data: t.List[t.Dict]) -> EvaluationDataset:
        samples = []
        if all(
            "user_input" in item and isinstance(data[0]["user_input"], list)
            for item in data
        ):
            samples.extend(MultiTurnSample(**sample) for sample in data)
        else:
            samples.extend(SingleTurnSample(**sample) for sample in data)
        return cls(samples=samples)

    def __repr__(self) -> str:
        return f"EvaluationDataset(features={self.features()}, len={len(self.samples)})"


@dataclass
class EvaluationResult:
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

    scores: t.List[t.Dict[str, t.Any]]
    dataset: EvaluationDataset
    binary_columns: t.List[str] = field(default_factory=list)
    cost_cb: t.Optional[CostCallbackHandler] = None
    traces: t.List[t.Dict[str, t.Any]] = field(default_factory=list)
    ragas_traces: t.Dict[str, ChainRun] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        # transform scores from list of dicts to dict of lists
        self._scores_dict = {
            k: [d[k] for d in self.scores] for k in self.scores[0].keys()
        }

        values = []
        self._repr_dict = {}
        for metric_name in self._scores_dict.keys():
            value = safe_nanmean(self._scores_dict[metric_name])
            self._repr_dict[metric_name] = value
            if metric_name not in self.binary_columns:
                value = t.cast(float, value)
                values.append(value + 1e-10)

        # parse the traces
        self.traces = parse_run_traces(self.ragas_traces)

    def __repr__(self) -> str:
        score_strs = [f"'{k}': {v:0.4f}" for k, v in self._repr_dict.items()]
        return "{" + ", ".join(score_strs) + "}"

    def __getitem__(self, key: str) -> t.List[float]:
        return self._scores_dict[key]

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
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is not installed. Please install it to use this function."
            )

        if self.dataset is None:
            raise ValueError("dataset is not provided for the results class")
        assert len(self.scores) == len(self.dataset)
        # convert both to pandas dataframes and concatenate
        scores_df = pd.DataFrame(self.scores)
        dataset_df = self.dataset.to_pandas()
        return pd.concat([dataset_df, scores_df], axis=1)

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

    def upload(self, base_url: str = RAGAS_API_URL, verbose: bool = True) -> str:
        from datetime import datetime, timezone

        import requests

        timestamp = datetime.now(timezone.utc).isoformat()
        root_trace = [
            trace for trace in self.ragas_traces.values() if trace.parent_run_id is None
        ][0]
        packet = json.dumps(
            {
                "run_id": str(root_trace.run_id),
                "created_at": timestamp,
                "evaluation_run": [t.model_dump() for t in self.ragas_traces.values()],
            },
            cls=ChainRunEncoder,
        )

        response = requests.post(
            f"{base_url}/alignment/evaluation",
            data=packet,
            headers={"Content-Type": "application/json"},
        )

        if response.status_code != 200:
            raise Exception(f"Failed to upload results: {response.text}")

        evaluation_endpoint = (
            f"https://app.ragas.io/alignment/evaluation/{root_trace.run_id}"
        )
        if verbose:
            print(f"Evaluation results uploaded! View at {evaluation_endpoint}")
        return evaluation_endpoint

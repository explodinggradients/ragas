"""
Q - question
A - answer: generated_text from RAG pipeline
C - contexts: context used for generation
G - ground_truth: ground truth answer
"""

from __future__ import annotations

import asyncio
import logging
import typing as t
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum

from ragas.callbacks import new_group
from ragas.dataset_schema import MultiTurnSample, SingleTurnSample
from ragas.executor import is_event_loop_running
from ragas.run_config import RunConfig
from ragas.utils import deprecated

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks

    from ragas.embeddings import BaseRagasEmbeddings
    from ragas.llms import BaseRagasLLM

import inspect

from pysbd import Segmenter
from pysbd.languages import LANGUAGE_CODES

from ragas.experimental.llms.prompt import PydanticPrompt as Prompt

logger = logging.getLogger(__name__)


LANGUAGE_CODES = {v.__name__.lower(): k for k, v in LANGUAGE_CODES.items()}

VALID_COLUMNS = [
    "user_input",
    "retrieved_contexts",
    "reference_contexts",
    "response",
    "reference",
    "rubric",
]


class MetricType(Enum):
    SINGLE_TURN = "single_turn"
    MULTI_TURN = "multi_turn"


@dataclass
class Metric(ABC):
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(default_factory=dict)

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    def required_columns(self) -> t.Dict[str, t.Set[str]]:
        return {k.name: v for k, v in self._required_columns.items()}

    @required_columns.setter
    def required_columns(self, metric_type: MetricType, columns: t.Set[str]):
        for column in columns:
            if column not in VALID_COLUMNS:
                raise ValueError(
                    f"Invalid column '{column}'. Must be one of {VALID_COLUMNS}"
                )
        self._required_columns[metric_type] = columns

    @abstractmethod
    def init(self, run_config: RunConfig):
        """
        This method will lazy initialize the model.
        """
        ...

    def adapt(self, language: str, cache_dir: t.Optional[str] = None) -> None:
        """
        Adapt the metric to a different language.
        """
        raise NotImplementedError(
            "adapt() is not implemented for {} metric".format(self.name)
        )

    def save(self, cache_dir: t.Optional[str] = None) -> None:
        """
        Save the metric to a path.
        """
        raise NotImplementedError(
            "adapt() is not implemented for {} metric".format(self.name)
        )

    @deprecated("0.2", removal="0.3", alternative="single_turn_ascore")
    def score(self: t.Self, row: t.Dict, callbacks: Callbacks = None) -> float:
        callbacks = callbacks or []
        rm, group_cm = new_group(self.name, inputs=row, callbacks=callbacks)
        try:
            if is_event_loop_running():
                try:
                    import nest_asyncio

                    nest_asyncio.apply()
                except ImportError:
                    raise ImportError(
                        "It seems like your running this in a jupyter-like environment. Please install nest_asyncio with `pip install nest_asyncio` to make it work."
                    )
            loop = asyncio.get_event_loop()
            score = loop.run_until_complete(self._ascore(row=row, callbacks=group_cm))
        except Exception as e:
            if not group_cm.ended:
                rm.on_chain_error(e)
            raise e
        else:
            if not group_cm.ended:
                rm.on_chain_end({"output": score})
        return score

    @deprecated("0.2", removal="0.3", alternative="single_turn_ascore")
    async def ascore(
        self: t.Self,
        row: t.Dict,
        callbacks: Callbacks = None,
        timeout: t.Optional[float] = None,
    ) -> float:
        callbacks = callbacks or []
        rm, group_cm = new_group(self.name, inputs=row, callbacks=callbacks)
        try:
            score = await asyncio.wait_for(
                self._ascore(row=row, callbacks=group_cm),
                timeout=timeout,
            )
        except Exception as e:
            if not group_cm.ended:
                rm.on_chain_error(e)
            raise e
        else:
            if not group_cm.ended:
                rm.on_chain_end({"output": score})
        return score

    @abstractmethod
    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        ...


@dataclass
class MetricWithLLM(Metric):
    llm: t.Optional[BaseRagasLLM] = None

    def init(self, run_config: RunConfig):
        """
        Init any models in the metric, this is invoked before evaluate()
        to load all the models
        Also check if the api key is valid for OpenAI and AzureOpenAI
        """
        if self.llm is None:
            raise ValueError(
                f"Metric '{self.name}' has no valid LLM provided (self.llm is None). Please initantiate a the metric with an LLM to run."  # noqa
            )
        self.llm.set_run_config(run_config)

    def get_prompts(self) -> t.Dict[str, Prompt]:
        prompts = {}
        for name, value in inspect.getmembers(self):
            if isinstance(value, Prompt):
                prompts.update({name: value})
        return prompts

    def set_prompts(self, **prompts):
        available_prompts = self.get_prompts()
        for key, value in prompts.items():
            if key not in available_prompts:
                raise ValueError(
                    f"Prompt with name '{key}' does not exist in the metric {self.name}. Use get_prompts() to see available prompts."
                )
            if not isinstance(value, Prompt):
                raise ValueError(
                    f"Prompt with name '{key}' must be an instance of 'Prompt'"
                )
            setattr(self, key, value)


@dataclass
class MetricWithEmbeddings(Metric):
    embeddings: t.Optional[BaseRagasEmbeddings] = None

    def init(self, run_config: RunConfig):
        """
        Init any models in the metric, this is invoked before evaluate()
        to load all the models
        Also check if the api key is valid for OpenAI and AzureOpenAI
        """
        if self.embeddings is None:
            raise ValueError(
                f"Metric '{self.name}' has no valid embeddings provided (self.embeddings is None). Please initantiate a the metric with an embeddings to run."  # noqa
            )
        self.embeddings.set_run_config(run_config)


class SingleTurnMetric(Metric):
    def single_turn_score(
        self,
        sample: SingleTurnSample,
        callbacks: Callbacks = None,
    ) -> float:
        callbacks = callbacks or []
        rm, group_cm = new_group(self.name, inputs=sample.dict(), callbacks=callbacks)
        try:
            loop = asyncio.get_event_loop()
            score = loop.run_until_complete(
                self._single_turn_ascore(sample=sample, callbacks=group_cm)
            )
        except Exception as e:
            if not group_cm.ended:
                rm.on_chain_error(e)
            raise e
        else:
            if not group_cm.ended:
                rm.on_chain_end({"output": score})
        return score

    async def single_turn_ascore(
        self,
        sample: SingleTurnSample,
        callbacks: Callbacks = None,
        timeout: t.Optional[float] = None,
    ) -> float:
        callbacks = callbacks or []
        row = sample.dict()
        rm, group_cm = new_group(self.name, inputs=row, callbacks=callbacks)
        try:
            score = await asyncio.wait_for(
                self._single_turn_ascore(sample=sample, callbacks=group_cm),
                timeout=timeout,
            )
        except Exception as e:
            if not group_cm.ended:
                rm.on_chain_error(e)
            raise e
        else:
            if not group_cm.ended:
                rm.on_chain_end({"output": score})
        return score

    @abstractmethod
    async def _single_turn_ascore(
        self,
        sample: SingleTurnSample,
        callbacks: Callbacks,
    ) -> float:
        ...


class MultiTurnMetric(Metric):
    def multi_turn_score(
        self,
        sample: MultiTurnSample,
        callbacks: Callbacks = None,
    ) -> float:
        callbacks = callbacks or []
        rm, group_cm = new_group(self.name, inputs=sample.dict(), callbacks=callbacks)
        try:
            loop = asyncio.get_event_loop()
            score = loop.run_until_complete(
                self._multi_turn_ascore(sample=sample, callbacks=group_cm)
            )
        except Exception as e:
            if not group_cm.ended:
                rm.on_chain_error(e)
            raise e
        else:
            if not group_cm.ended:
                rm.on_chain_end({"output": score})
        return score

    async def multi_turn_ascore(
        self,
        sample: MultiTurnSample,
        callbacks: Callbacks = None,
        timeout: t.Optional[float] = None,
    ) -> float:
        callbacks = callbacks or []
        rm, group_cm = new_group(self.name, inputs=sample.dict(), callbacks=callbacks)
        try:
            score = await asyncio.wait_for(
                self._multi_turn_ascore(sample=sample, callbacks=group_cm),
                timeout=timeout,
            )
        except Exception as e:
            if not group_cm.ended:
                rm.on_chain_error(e)
            raise e
        else:
            if not group_cm.ended:
                rm.on_chain_end({"output": score})
        return score

    @abstractmethod
    async def _multi_turn_ascore(
        self,
        sample: MultiTurnSample,
        callbacks: Callbacks,
    ) -> float:
        ...


class Ensember:
    """
    Combine multiple llm outputs for same input (n>1) to a single output
    """

    def from_discrete(self, inputs: list[list[t.Dict]], attribute: str):
        """
        Simple majority voting for binary values, ie [0,0,1] -> 0
        inputs: list of list of dicts each containing verdict for a single input
        """

        if not isinstance(inputs, list):
            inputs = [inputs]

        if not all(len(item) == len(inputs[0]) for item in inputs):
            logger.warning("All inputs must have the same length")
            return inputs[0]

        if not all(attribute in item for input in inputs for item in input):
            logger.warning(f"All inputs must have {attribute} attribute")
            return inputs[0]

        if len(inputs) == 1:
            return inputs[0]

        verdict_agg = []
        for i in range(len(inputs[0])):
            item = inputs[0][i]
            verdicts = [inputs[k][i][attribute] for k in range(len(inputs))]
            verdict_counts = dict(Counter(verdicts).most_common())
            item[attribute] = list(verdict_counts.keys())[0]
            verdict_agg.append(item)

        return verdict_agg


def get_segmenter(
    language: str = "english", clean: bool = False, char_span: bool = False
):
    """
    Get a sentence segmenter for a given language
    """
    language = language.lower()
    if language not in LANGUAGE_CODES:
        raise ValueError(
            f"Language '{language}' not supported. Supported languages: {LANGUAGE_CODES.keys()}"
        )
    return Segmenter(
        language=LANGUAGE_CODES[language], clean=clean, char_span=char_span
    )


def is_reproducable(metric: Metric) -> bool:
    return hasattr(metric, "_reproducibility")


ensembler = Ensember()

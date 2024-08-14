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
from dataclasses import dataclass
from enum import Enum

from ragas.callbacks import new_group
from ragas.run_config import RunConfig

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks

    from ragas.embeddings import BaseRagasEmbeddings
    from ragas.llms import BaseRagasLLM

from pysbd import Segmenter
from pysbd.languages import LANGUAGE_CODES

logger = logging.getLogger(__name__)


LANGUAGE_CODES = {v.__name__.lower(): k for k, v in LANGUAGE_CODES.items()}

EvaluationMode = Enum("EvaluationMode", "qac qa qc gc ga qga qcg ca")


def get_required_columns(
    eval_mod: EvaluationMode, ignore_columns: t.Optional[t.List[str]] = None
) -> t.List[str]:
    if eval_mod == EvaluationMode.qac:
        keys = ["question", "answer", "contexts"]
    elif eval_mod == EvaluationMode.qa:
        keys = ["question", "answer"]
    elif eval_mod == EvaluationMode.qc:
        keys = ["question", "contexts"]
    elif eval_mod == EvaluationMode.gc:
        keys = ["contexts", "ground_truth"]
    elif eval_mod == EvaluationMode.ga:
        keys = ["answer", "ground_truth"]
    elif eval_mod == EvaluationMode.qga:
        keys = ["question", "contexts", "answer", "ground_truth"]
    elif eval_mod == EvaluationMode.qcg:
        keys = ["question", "contexts", "ground_truth"]
    elif eval_mod == EvaluationMode.ca:
        keys = ["contexts", "answer"]
    ignore_columns = ignore_columns or []

    return [k for k in keys if k not in ignore_columns]


@dataclass
class Metric(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def evaluation_mode(self) -> EvaluationMode:
        ...

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

    def score(self: t.Self, row: t.Dict, callbacks: Callbacks = None) -> float:
        callbacks = callbacks or []
        rm, group_cm = new_group(self.name, inputs=row, callbacks=callbacks)
        try:
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

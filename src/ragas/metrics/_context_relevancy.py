from __future__ import annotations

import logging
import typing as t
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pysbd
from datasets import Dataset
from langchain.callbacks.manager import CallbackManager, trace_as_chain_group

from ragas.llms.prompt import Prompt
from ragas.metrics.base import EvaluationMode, MetricWithLLM

if t.TYPE_CHECKING:
    from langchain.callbacks.base import Callbacks

logger = logging.getLogger(__name__)

CONTEXT_RELEVANCE = Prompt(
    name="context_relevancy",
    instruction="""Please extract relevant sentences from the provided context that is absolutely required answer the following question. If no relevant sentences are found, or if you believe the question cannot be answered from the given context, return the phrase "Insufficient Information".  While extracting candidate sentences you're not allowed to make any changes to sentences from given context.""",
    input_keys=["question", "context"],
    output_key="candidate sentences",
    output_type="json",
)


seg = pysbd.Segmenter(language="en", clean=False)


def sent_tokenize(text: str) -> List[str]:
    """
    tokenizer text into sentences
    """
    sentences = seg.segment(text)
    assert isinstance(sentences, list)
    return sentences


@dataclass
class ContextRelevancy(MetricWithLLM):
    """
    Extracts sentences from the context that are relevant to the question with
    self-consistancy checks. The number of relevant sentences and is used as the score.

    Attributes
    ----------
    name : str
    batch_size : int
        Batch size for openai completion.
    """

    name: str = "context_relevancy"  # type: ignore
    evaluation_mode: EvaluationMode = EvaluationMode.qc  # type: ignore
    context_relevancy_prompt: Prompt = field(default_factory=lambda: CONTEXT_RELEVANCE)
    batch_size: int = 15
    show_deprecation_warning: bool = False

    def adapt(self, language: str, cache_dir: str | None = None) -> None:
        logger.info(f"Adapting Context Relevancy to {language}")
        self.context_relevancy_prompt = self.context_relevancy_prompt.adapt(
            language, self.llm, cache_dir
        )

    def save(self, cache_dir: str | None = None) -> None:
        self.context_relevancy_prompt.save(cache_dir)

    def _score_batch(
        self: t.Self,
        dataset: Dataset,
        callbacks: t.Optional[Callbacks] = None,
        callback_group_name: str = "batch",
    ) -> list[float]:
        if self.show_deprecation_warning:
            logger.warning(
                "The 'context_relevancy' metric is going to be deprecated soon! Please use the 'context_precision' metric instead. It is a drop-in replacement just a simple search and replace should work."  # noqa
            )
        prompts = []
        questions, contexts = dataset["question"], dataset["contexts"]

        cb = CallbackManager.configure(inheritable_callbacks=callbacks)
        with trace_as_chain_group(
            callback_group_name, callback_manager=cb
        ) as batch_group:
            for q, c in zip(questions, contexts):
                prompts.append(
                    self.context_relevancy_prompt.format(
                        question=q, context="\n".join(c)
                    )
                )

            responses: list[list[str]] = []
            results = self.llm.generate(
                prompts,
                n=1,
                callbacks=batch_group,
            )
            responses = [[i.text for i in r] for r in results.generations]

            scores = []
            for context, n_response in zip(contexts, responses):
                context = "\n".join(context)
                overlap_scores = []
                context_sents = sent_tokenize(context)
                for output in n_response:
                    indices = (
                        sent_tokenize(output.strip())
                        if output.lower() != "insufficient information."
                        else []
                    )
                    if len(context_sents) == 0:
                        score = 0
                    else:
                        score = min(len(indices) / len(context_sents), 1)
                    overlap_scores.append(score)
                scores.append(np.mean(overlap_scores))

        return scores


context_relevancy = ContextRelevancy()

from __future__ import annotations

import logging
import typing as t
from dataclasses import dataclass, field
from typing import List

import pysbd

from ragas.llms.prompt import Prompt
from ragas.metrics.base import EvaluationMode, MetricWithLLM

if t.TYPE_CHECKING:
    from langchain_core.callbacks.base import Callbacks

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
    self-consistency checks. The number of relevant sentences and is used as the score.

    Attributes
    ----------
    name : str
    """

    name: str = "context_relevancy"  # type: ignore
    evaluation_mode: EvaluationMode = EvaluationMode.qc  # type: ignore
    context_relevancy_prompt: Prompt = field(default_factory=lambda: CONTEXT_RELEVANCE)
    show_deprecation_warning: bool = False

    def _compute_score(self, response: str, row: t.Dict) -> float:
        context = "\n".join(row["contexts"])
        context_sents = sent_tokenize(context)
        indices = (
            sent_tokenize(response.strip())
            if response.lower() != "insufficient information."
            else []
        )
        # print(len(indices))
        if len(context_sents) == 0:
            return 0
        else:
            return min(len(indices) / len(context_sents), 1)

    async def _ascore(self, row: t.Dict, callbacks: Callbacks, is_async: bool) -> float:
        assert self.llm is not None, "LLM is not initialized"

        if self.show_deprecation_warning:
            logger.warning(
                "The 'context_relevancy' metric is going to be deprecated soon! Please use the 'context_precision' metric instead. It is a drop-in replacement just a simple search and replace should work."  # noqa
            )

        question, contexts = row["question"], row["contexts"]
        result = await self.llm.generate(
            self.context_relevancy_prompt.format(
                question=question, context="\n".join(contexts)
            ),
            callbacks=callbacks,
        )
        return self._compute_score(result.generations[0][0].text, row)

    def adapt(self, language: str, cache_dir: str | None = None) -> None:
        assert self.llm is not None, "set LLM before use"

        logger.info(f"Adapting Context Relevancy to {language}")
        self.context_relevancy_prompt = self.context_relevancy_prompt.adapt(
            language, self.llm, cache_dir
        )

    def save(self, cache_dir: str | None = None) -> None:
        self.context_relevancy_prompt.save(cache_dir)


context_relevancy = ContextRelevancy()

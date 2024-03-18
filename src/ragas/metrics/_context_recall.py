from __future__ import annotations

import logging
import typing as t
from dataclasses import dataclass, field

import numpy as np

from ragas.llms.json_load import json_loader
from ragas.llms.prompt import Prompt
from ragas.metrics.base import EvaluationMode, MetricWithLLM

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks

    from ragas.llms.prompt import PromptValue

logger = logging.getLogger(__name__)

CONTEXT_RECALL_RA = Prompt(
    name="context_recall",
    instruction="""Given a context, and an answer, analyze each sentence in the answer and classify if the sentence can be attributed to the given context or not. Use only "Yes" (1) or "No" (0) as a binary classification. Output json with reason.""",
    examples=[
        {
            "question": """What can you tell me about albert Albert Einstein?""",
            "context": """Albert Einstein (14 March 1879 – 18 April 1955) was a German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time. Best known for developing the theory of relativity, he also made important contributions to quantum mechanics, and was thus a central figure in the revolutionary reshaping of the scientific understanding of nature that modern physics accomplished in the first decades of the twentieth century. His mass–energy equivalence formula E = mc2, which arises from relativity theory, has been called 'the world's most famous equation'. He received the 1921 Nobel Prize in Physics 'for his services to theoretical physics, and especially for his discovery of the law of the photoelectric effect', a pivotal step in the development of quantum theory. His work is also known for its influence on the philosophy of science. In a 1999 poll of 130 leading physicists worldwide by the British journal Physics World, Einstein was ranked the greatest physicist of all time. His intellectual achievements and originality have made Einstein synonymous with genius.""",
            "answer": """Albert Einstein born in 14 March 1879 was  German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time. He received the 1921 Nobel Prize in Physics for his services to theoretical physics. He published 4 papers in 1905.  Einstein moved to Switzerland in 1895""",
            "classification": [
                {
                    "statement_1": "Albert Einstein, born on 14 March 1879, was a German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time.",
                    "reason": "The date of birth of Einstein is mentioned clearly in the context.",
                    "Attributed": "1",
                },
                {
                    "statement_2": "He received the 1921 Nobel Prize in Physics 'for his services to theoretical physics.",
                    "reason": "The exact sentence is present in the given context.",
                    "Attributed": "1",
                },
                {
                    "statement_3": "He published 4 papers in 1905.",
                    "reason": "There is no mention about papers he wrote in the given context.",
                    "Attributed": "0",
                },
                {
                    "statement_4": "Einstein moved to Switzerland in 1895.",
                    "reason": "There is no supporting evidence for this in the given context.",
                    "Attributed": "0",
                },
            ],
        },
        {
            "question": """who won 2020 icc world cup?""",
            "context": """The 2022 ICC Men's T20 World Cup, held from October 16 to November 13, 2022, in Australia, was the eighth edition of the tournament. Originally scheduled for 2020, it was postponed due to the COVID-19 pandemic. England emerged victorious, defeating Pakistan by five wickets in the final to clinch their second ICC Men's T20 World Cup title.""",
            "answer": """England""",
            "classification": {
                "statement_1": "England won the 2022 ICC Men's T20 World Cup.",
                "reason": "From context it is clear that England defeated Pakistan to win the World Cup.",
                "Attributed": "1",
            },
        },
        {
            "question": """What is the primary fuel for the Sun?""",
            "context": """NULL""",
            "answer": """Hydrogen""",
            "classification": {
                "statement_1": "The Sun's primary fuel is hydrogen.",
                "reason": "The context contains no information",
                "Attributed": "0",
            },
        },
    ],
    input_keys=["question", "context", "answer"],
    output_key="classification",
    output_type="json",
)


@dataclass
class ContextRecall(MetricWithLLM):

    """
    Estimates context recall by estimating TP and FN using annotated answer and
    retrieved context.

    Attributes
    ----------
    name : str
    """

    name: str = "context_recall"  # type: ignore
    evaluation_mode: EvaluationMode = EvaluationMode.qcg  # type: ignore
    context_recall_prompt: Prompt = field(default_factory=lambda: CONTEXT_RECALL_RA)

    def _create_context_recall_prompt(self, row: t.Dict) -> PromptValue:
        qstn, ctx, gt = row["question"], row["contexts"], row["ground_truth"]
        ctx = "\n".join(ctx) if isinstance(ctx, list) else ctx

        return self.context_recall_prompt.format(question=qstn, context=ctx, answer=gt)

    def _compute_score(self, response: t.Any) -> float:
        response = response if isinstance(response, list) else [response]
        response = [item if isinstance(item, dict) else {} for item in response]
        response = [
            int(item.get("Attributed").strip() == "1")
            if item.get("Attributed")
            else np.nan
            for item in response
        ]
        denom = len(response)
        numerator = sum(response)
        score = numerator / denom if denom > 0 else np.nan

        if np.isnan(score):
            logger.warning(
                "Invalid JSON response. Expected dictionary with key 'Attributed'"
            )

        return score

    async def _ascore(self, row: t.Dict, callbacks: Callbacks, is_async: bool) -> float:
        assert self.llm is not None, "set LLM before use"

        result = await self.llm.generate(
            self._create_context_recall_prompt(row), callbacks=callbacks, is_async=is_async
        )
        response = await json_loader.safe_load(
            result.generations[0][0].text, self.llm, is_async=is_async
        )

        return self._compute_score(response)

    def adapt(self, language: str, cache_dir: str | None = None) -> None:
        assert self.llm is not None, "set LLM before use"

        logger.info(f"Adapting Context Recall to {language}")
        self.context_recall_prompt = self.context_recall_prompt.adapt(
            language, self.llm, cache_dir
        )

    def save(self, cache_dir: str | None = None) -> None:
        self.context_recall_prompt.save(cache_dir)


context_recall = ContextRecall()

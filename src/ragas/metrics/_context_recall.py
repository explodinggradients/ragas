from __future__ import annotations

import logging
import typing as t
from dataclasses import dataclass, field

import numpy as np
from langchain_core.pydantic_v1 import BaseModel

from ragas.dataset_schema import SingleTurnSample
from ragas.llms.output_parser import RagasoutputParser, get_json_format_instructions
from ragas.llms.prompt import Prompt
from ragas.metrics._string import NonLLMStringSimilarity
from ragas.metrics.base import MetricType, MetricWithLLM, SingleTurnMetric, ensembler
from ragas.run_config import RunConfig
from ragas.utils import deprecated

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks

    from ragas.llms.prompt import PromptValue

logger = logging.getLogger(__name__)


class ContextRecallClassificationAnswer(BaseModel):
    statement: str
    attributed: int
    reason: str


class ContextRecallClassificationAnswers(BaseModel):
    __root__: t.List[ContextRecallClassificationAnswer]

    def dicts(self) -> t.List[t.Dict]:
        return self.dict()["__root__"]


_classification_output_instructions = get_json_format_instructions(
    ContextRecallClassificationAnswers
)
_output_parser = RagasoutputParser(pydantic_object=ContextRecallClassificationAnswers)


CONTEXT_RECALL_RA = Prompt(
    name="context_recall",
    instruction="""Given a context, and an answer, analyze each sentence in the answer and classify if the sentence can be attributed to the given context or not. Use only "Yes" (1) or "No" (0) as a binary classification. Output json with reason.""",
    output_format_instruction=_classification_output_instructions,
    examples=[
        {
            "question": """What can you tell me about albert Albert Einstein?""",
            "context": """Albert Einstein (14 March 1879 - 18 April 1955) was a German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time. Best known for developing the theory of relativity, he also made important contributions to quantum mechanics, and was thus a central figure in the revolutionary reshaping of the scientific understanding of nature that modern physics accomplished in the first decades of the twentieth century. His mass-energy equivalence formula E = mc2, which arises from relativity theory, has been called 'the world's most famous equation'. He received the 1921 Nobel Prize in Physics 'for his services to theoretical physics, and especially for his discovery of the law of the photoelectric effect', a pivotal step in the development of quantum theory. His work is also known for its influence on the philosophy of science. In a 1999 poll of 130 leading physicists worldwide by the British journal Physics World, Einstein was ranked the greatest physicist of all time. His intellectual achievements and originality have made Einstein synonymous with genius.""",
            "answer": """Albert Einstein born in 14 March 1879 was  German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time. He received the 1921 Nobel Prize in Physics for his services to theoretical physics. He published 4 papers in 1905.  Einstein moved to Switzerland in 1895""",
            "classification": ContextRecallClassificationAnswers.parse_obj(
                [
                    {
                        "statement": "Albert Einstein, born on 14 March 1879, was a German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time.",
                        "reason": "The date of birth of Einstein is mentioned clearly in the context.",
                        "attributed": 1,
                    },
                    {
                        "statement": "He received the 1921 Nobel Prize in Physics for his services to theoretical physics.",
                        "reason": "The exact sentence is present in the given context.",
                        "attributed": 1,
                    },
                    {
                        "statement": "He published 4 papers in 1905.",
                        "reason": "There is no mention about papers he wrote in the given context.",
                        "attributed": 0,
                    },
                    {
                        "statement": "Einstein moved to Switzerland in 1895.",
                        "reason": "There is no supporting evidence for this in the given context.",
                        "attributed": 0,
                    },
                ]
            ).dicts(),
        },
        {
            "question": """who won 2020 icc world cup?""",
            "context": """The 2022 ICC Men's T20 World Cup, held from October 16 to November 13, 2022, in Australia, was the eighth edition of the tournament. Originally scheduled for 2020, it was postponed due to the COVID-19 pandemic. England emerged victorious, defeating Pakistan by five wickets in the final to clinch their second ICC Men's T20 World Cup title.""",
            "answer": """England""",
            "classification": ContextRecallClassificationAnswers.parse_obj(
                [
                    {
                        "statement": "England won the 2022 ICC Men's T20 World Cup.",
                        "reason": "From context it is clear that England defeated Pakistan to win the World Cup.",
                        "attributed": 1,
                    },
                ]
            ).dicts(),
        },
        {
            "question": """What is the primary fuel for the Sun?""",
            "context": """NULL""",
            "answer": """Hydrogen""",
            "classification": ContextRecallClassificationAnswers.parse_obj(
                [
                    {
                        "statement": "The Sun's primary fuel is hydrogen.",
                        "reason": "The context contains no information",
                        "attributed": 0,
                    },
                ]
            ).dicts(),
        },
    ],
    input_keys=["question", "context", "answer"],
    output_key="classification",
    output_type="json",
)


@dataclass
class LLMContextRecall(MetricWithLLM, SingleTurnMetric):
    """
    Estimates context recall by estimating TP and FN using annotated answer and
    retrieved context.

    Attributes
    ----------
    name : str
    """

    name: str = "context_recall"  # type: ignore
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {
                "user_input",
                "retrieved_contexts",
                "reference",
            }
        }
    )
    context_recall_prompt: Prompt = field(default_factory=lambda: CONTEXT_RECALL_RA)
    max_retries: int = 1
    _reproducibility: int = 1

    @property
    def reproducibility(self):
        return self._reproducibility

    @reproducibility.setter
    def reproducibility(self, value):
        if value < 1:
            logger.warning("reproducibility cannot be less than 1, setting to 1")
            value = 1
        elif value % 2 == 0:
            logger.warning(
                "reproducibility level cannot be set to even number, setting to odd"
            )
            value += 1
        self._reproducibility = value

    def __post_init__(self) -> None:
        if self.reproducibility < 1:
            logger.warning("reproducibility cannot be less than 1, setting to 1")
            self.reproducibility = 1

    def _create_context_recall_prompt(self, row: t.Dict) -> PromptValue:
        qstn, ctx, gt = row["user_input"], row["retrieved_contexts"], row["reference"]
        ctx = "\n".join(ctx) if isinstance(ctx, list) else ctx

        return self.context_recall_prompt.format(question=qstn, context=ctx, answer=gt)

    def _compute_score(self, response: t.Any) -> float:
        response = [1 if item.attributed else 0 for item in response.__root__]
        denom = len(response)
        numerator = sum(response)
        score = numerator / denom if denom > 0 else np.nan

        if np.isnan(score):
            logger.warning("The LLM did not return a valid classification.")

        return score

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        row = sample.dict()
        return await self._ascore(row, callbacks)

    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        assert self.llm is not None, "set LLM before use"
        p_value = self._create_context_recall_prompt(row)
        results = await self.llm.generate(
            p_value,
            callbacks=callbacks,
            n=self.reproducibility,
        )
        results = [results.generations[0][i].text for i in range(self.reproducibility)]

        answers = [
            await _output_parser.aparse(text, p_value, self.llm, self.max_retries)
            for text in results
        ]

        answers = [answer.dicts() for answer in answers if answer is not None]
        if all(answer is None for answer in answers):
            return np.nan

        answers = ensembler.from_discrete(answers, "attributed")
        answers = ContextRecallClassificationAnswers.parse_obj(answers)

        return self._compute_score(answers)

    def adapt(self, language: str, cache_dir: str | None = None) -> None:
        assert self.llm is not None, "set LLM before use"

        logger.info(f"Adapting Context Recall to {language}")
        self.context_recall_prompt = self.context_recall_prompt.adapt(
            language, self.llm, cache_dir
        )

    def save(self, cache_dir: str | None = None) -> None:
        self.context_recall_prompt.save(cache_dir)


@dataclass
class ContextRecall(LLMContextRecall):
    name: str = "context_recall"

    @deprecated(since="0.2", removal="0.3", alternative="LLMContextRecall")
    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        row = sample.dict()
        return await self._ascore(row, callbacks)

    @deprecated(since="0.2", removal="0.3", alternative="LLMContextRecall")
    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        return await super()._ascore(row, callbacks)


@dataclass
class NonLLMContextRecall(SingleTurnMetric):
    name: str = "non_llm_context_recall"  # type: ignore
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {
                "retrieved_contexts",
                "reference_contexts",
            }
        }
    )
    distance_measure: SingleTurnMetric = field(
        default_factory=lambda: NonLLMStringSimilarity()
    )
    threshold: float = 0.5

    def __post_init__(self):
        if isinstance(self.distance_measure, MetricWithLLM):
            raise ValueError(
                "distance_measure must not be an instance of MetricWithLLM for NonLLMContextPrecisionWithReference"
            )

    def init(self, run_config: RunConfig) -> None:
        ...

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        retrieved_contexts = sample.retrieved_contexts
        reference_contexts = sample.reference_contexts
        assert retrieved_contexts is not None, "retrieved_contexts is empty"
        assert reference_contexts is not None, "reference_contexts is empty"

        scores = []
        for ref in reference_contexts:
            scores.append(
                max(
                    [
                        await self.distance_measure.single_turn_ascore(
                            SingleTurnSample(reference=rc, response=ref), callbacks
                        )
                        for rc in retrieved_contexts
                    ]
                )
            )
        return self._compute_score(scores)

    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        return await self._single_turn_ascore(SingleTurnSample(**row), callbacks)

    def _compute_score(self, verdict_list: t.List[float]) -> float:
        response = [1 if score > self.threshold else 0 for score in verdict_list]
        denom = len(response)
        numerator = sum(response)
        score = numerator / denom if denom > 0 else np.nan
        return score


context_recall = ContextRecall()

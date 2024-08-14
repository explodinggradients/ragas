from __future__ import annotations

import inspect
import json
import logging
import typing as t
from dataclasses import dataclass, field

import numpy as np

from ragas.llms.prompt import Prompt
from ragas.metrics._faithfulness import (
    LONG_FORM_ANSWER_PROMPT,
    NLI_STATEMENTS_MESSAGE,
    HasSegmentMethod,
    StatementFaithfulnessAnswers,
    _faithfulness_output_parser,
    _statements_output_parser,
)
from ragas.metrics.base import EvaluationMode, MetricWithLLM, ensembler, get_segmenter

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks

    from ragas.llms.prompt import PromptValue


logger = logging.getLogger(__name__)


@dataclass
class NoiseSensitivity(MetricWithLLM):
    name: str = "noise_sensitivity"  # type: ignore
    evaluation_mode: EvaluationMode = EvaluationMode.qga  # type: ignore
    nli_statements_message: Prompt = field(
        default_factory=lambda: NLI_STATEMENTS_MESSAGE
    )
    statement_prompt: Prompt = field(default_factory=lambda: LONG_FORM_ANSWER_PROMPT)
    sentence_segmenter: t.Optional[HasSegmentMethod] = None
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

    def __post_init__(self):
        if self.sentence_segmenter is None:
            language = self.nli_statements_message.language
            self.sentence_segmenter = get_segmenter(language=language, clean=False)

    def _create_nli_prompt(self, contexts: str, statements: t.List[str]) -> PromptValue:
        assert self.llm is not None, "llm must be set to compute score"

        statements_str: str = json.dumps(statements)
        prompt_value = self.nli_statements_message.format(
            context=contexts, statements=statements_str
        )
        return prompt_value

    def _create_statements_prompt(self, text: str, question: str) -> PromptValue:
        assert self.sentence_segmenter is not None, "sentence_segmenter is not set"
        # contexts = row["contexts"]
        sentences = self.sentence_segmenter.segment(text)
        sentences = [
            sentence for sentence in sentences if sentence.strip().endswith(".")
        ]
        sentences = "\n".join([f"{i}:{x}" for i, x in enumerate(sentences)])
        prompt_value = self.statement_prompt.format(
            question=question, answer=text, sentences=sentences
        )
        return prompt_value

    async def _evaluate_statement_faithfulness(
        self, statements, context: str, callbacks: Callbacks
    ):
        assert self.llm is not None, "LLM is not set"

        p_value = self._create_nli_prompt(context, statements)
        nli_result = await self.llm.generate(
            p_value,
            callbacks=callbacks,
            n=self._reproducibility,
        )

        nli_result_text = [
            nli_result.generations[0][i].text for i in range(self._reproducibility)
        ]
        faithfulness_list = [
            await _faithfulness_output_parser.aparse(
                text, p_value, self.llm, self.max_retries
            )
            for text in nli_result_text
        ]

        faithfulness_list = [
            faith.dicts() for faith in faithfulness_list if faith is not None
        ]

        if faithfulness_list:
            faithfulness_list = ensembler.from_discrete(
                faithfulness_list,
                "verdict",
            )

            faithfulness_list = StatementFaithfulnessAnswers.parse_obj(
                faithfulness_list
            )

            verdict_list = [
                1 if statement.verdict else 0
                for statement in faithfulness_list.__root__
            ]
            return np.array(verdict_list)
        else:
            return np.nan

    async def _decompose_answer_into_statements(
        self, text: str, question: str, callbacks: Callbacks
    ):
        assert self.llm is not None, "LLM is not set"

        p_value = self._create_statements_prompt(text, question)

        if inspect.iscoroutinefunction(self.llm.generate):
            statements_gen = await self.llm.generate(
                p_value,
                callbacks=callbacks,
            )
        else:
            statements_gen = self.llm.generate(
                p_value,
                callbacks=callbacks,
            )

        # Await the aparse method
        statements = await _statements_output_parser.aparse(
            statements_gen.generations[0][0].text, p_value, self.llm, self.max_retries  # type: ignore
        )

        if statements is None:
            return np.nan

        # Ensure statements is not a coroutine before calling dicts()
        if inspect.iscoroutine(statements):
            statements = await statements

        # Add error handling and logging
        if not hasattr(statements, "dicts"):
            logging.error(f"Unexpected type for statements: {type(statements)}")
            logging.error(f"Statements content: {statements}")
            raise AttributeError(
                f"'statements' object of type {type(statements)} has no attribute 'dicts'"
            )

        statements = [item["simpler_statements"] for item in statements.dicts()]
        statements = [item for sublist in statements for item in sublist]

        return statements

    def _compute_score(self, answers: t.Dict):
        # check the verdicts and compute the score
        relevant_retrieved = np.max(answers["retrieved2answer"], axis=0, keepdims=True)
        relevant_faithful = np.max(
            relevant_retrieved & answers["retrieved2response"], axis=1
        )
        incorrect = ~answers["answer2response"]

        noise_sensitivity_in_relevant = np.mean(relevant_faithful & incorrect)
        return noise_sensitivity_in_relevant

    async def _ascore(self: t.Self, row: t.Dict, callbacks: Callbacks) -> float:
        """
        returns the NLI score for each (q, c, a) pair
        """
        assert self.llm is not None, "LLM is not set"

        gt_statements = await self._decompose_answer_into_statements(
            row["ground_truth"], row["question"], callbacks
        )
        ans_statements = await self._decompose_answer_into_statements(
            row["answer"], row["question"], callbacks
        )

        answers_verdictslist = []
        response_verdictslist = []

        for ctx in row["contexts"]:
            verdicts = await self._evaluate_statement_faithfulness(
                gt_statements, ctx, callbacks
            )
            answers_verdictslist.append(verdicts)

            verdicts = await self._evaluate_statement_faithfulness(
                ans_statements, ctx, callbacks
            )
            response_verdictslist.append(verdicts)

        answers = {}
        answers["retrieved2answer"] = np.array(answers_verdictslist).T
        answers["retrieved2response"] = np.array(response_verdictslist).T
        answers["answer2response"] = await self._evaluate_statement_faithfulness(
            ans_statements, row["ground_truth"], callbacks
        )
        return self._compute_score(answers)

    def adapt(self, language: str, cache_dir: t.Optional[str] = None) -> None:
        assert self.llm is not None, "LLM is not set"

        logger.info(f"Adapting Faithfulness metric to {language}")

        self.nli_statements_message = self.nli_statements_message.adapt(
            language, self.llm, cache_dir
        )
        self.statement_prompt = self.statement_prompt.adapt(
            language, self.llm, cache_dir
        )

        self.sentence_segmenter = get_segmenter(language=language, clean=False)

    def save(self, cache_dir: t.Optional[str] = None) -> None:
        self.nli_statements_message.save(cache_dir)
        self.statement_prompt.save(cache_dir)


noise_sensitivity = NoiseSensitivity()

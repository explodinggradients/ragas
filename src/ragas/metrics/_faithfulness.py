from __future__ import annotations

import json
import logging
import typing as t
from dataclasses import dataclass, field

import numpy as np
from langchain_core.pydantic_v1 import BaseModel, Field

from ragas.llms.output_parser import RagasoutputParser, get_json_format_instructions
from ragas.llms.prompt import Prompt
from ragas.metrics.base import EvaluationMode, MetricWithLLM

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks

    from ragas.llms.prompt import PromptValue

logger = logging.getLogger(__name__)


class StatementsAnswers(BaseModel):
    __root__: t.List[str] = Field(..., description="the list of extracted statements")

    def dicts(self) -> t.List[t.Dict]:
        return self.dict()["__root__"]


_statements_output_instructions = get_json_format_instructions(StatementsAnswers)
_statements_output_parser = RagasoutputParser(pydantic_object=StatementsAnswers)


LONG_FORM_ANSWER_PROMPT = Prompt(
    name="long_form_answer",
    instruction="Create one or more statements from each sentence in the given answer.",
    output_format_instruction=_statements_output_instructions,
    examples=[
        {
            "question": "Who was  Albert Einstein and what is he best known for?",
            "answer": "He was a German-born theoretical physicist, widely acknowledged to be one of the greatest and most influential physicists of all time. He was best known for developing the theory of relativity, he also made important contributions to the development of the theory of quantum mechanics.",
            "statements": StatementsAnswers.parse_obj(
                [
                    "Albert Einstein, a German-born theoretical physicist, is renowned for being one of the most influential physicists in history.",
                    "Albert Einstein was best known for his theory of relativity.",
                    "Einstein's contributions significantly advanced the field of quantum mechanics",
                    "Recognized globally, Einstein's work has profoundly impacted the scientific community",
                    "Einstein's groundbreaking theories continue to shape our understanding of physics today.",
                ]
            ).dicts(),
        },
        {
            "question": "Cadmium Chloride is slightly soluble in this chemical, it is also called what?",
            "answer": "alcohol",
            "statements": StatementsAnswers.parse_obj(
                ["Cadmium Chloride is slightly soluble in alcohol."]
            ).dicts(),
        },
        {
            "question": "Were Hitler and Benito Mussolini of the same nationality?",
            "answer": "Sorry, I can't provide answer to that question.",
            "statements": StatementsAnswers.parse_obj([]).dicts(),
        },
    ],
    input_keys=["question", "answer"],
    output_key="statements",
    output_type="json",
)  # noqa: E501


class StatementFaithfulnessAnswer(BaseModel):
    statement: str = Field(..., description="the original statement, word-by-word")
    verdict: int = Field(..., description="the verdict(0/1) of the faithfulness.")
    reason: str = Field(..., description="the reason of the verdict")


class StatementFaithfulnessAnswers(BaseModel):
    __root__: t.List[StatementFaithfulnessAnswer]

    def dicts(self) -> t.List[t.Dict]:
        return self.dict()["__root__"]


_faithfulness_output_instructions = get_json_format_instructions(
    StatementFaithfulnessAnswers
)
_faithfulness_output_parser = RagasoutputParser(
    pydantic_object=StatementFaithfulnessAnswers
)

NLI_STATEMENTS_MESSAGE = Prompt(
    name="nli_statements",
    instruction="Your task is to judge the faithfulness of a series of statements based on a given context. For each statement you must return verdict as 1 if the statement can be verified based on the context or 0 if the statement can not be verified based on the context.",
    output_format_instruction=_faithfulness_output_instructions,
    examples=[
        {
            "context": """John is a student at XYZ University. He is pursuing a degree in Computer Science. He is enrolled in several courses this semester, including Data Structures, Algorithms, and Database Management. John is a diligent student and spends a significant amount of time studying and completing assignments. He often stays late in the library to work on his projects.""",
            "statements": [
                "John is majoring in Biology.",
                "John is taking a course on Artificial Intelligence.",
                "John is a dedicated student.",
                "John has a part-time job.",
            ],
            "answer": StatementFaithfulnessAnswers.parse_obj(
                [
                    {
                        "statement": "John is majoring in Biology.",
                        "reason": "John's major is explicitly mentioned as Computer Science. There is no information suggesting he is majoring in Biology.",
                        "verdict": 0,
                    },
                    {
                        "statement": "John is taking a course on Artificial Intelligence.",
                        "reason": "The context mentions the courses John is currently enrolled in, and Artificial Intelligence is not mentioned. Therefore, it cannot be deduced that John is taking a course on AI.",
                        "verdict": 0,
                    },
                    {
                        "statement": "John is a dedicated student.",
                        "reason": "The context states that he spends a significant amount of time studying and completing assignments. Additionally, it mentions that he often stays late in the library to work on his projects, which implies dedication.",
                        "verdict": 1,
                    },
                    {
                        "statement": "John has a part-time job.",
                        "reason": "There is no information given in the context about John having a part-time job.",
                        "verdict": 0,
                    },
                ]
            ).dicts(),
        },
        {
            "context": """Photosynthesis is a process used by plants, algae, and certain bacteria to convert light energy into chemical energy.""",
            "statements": ["Albert Einstein was a genius."],
            "answer": StatementFaithfulnessAnswers.parse_obj(
                [
                    {
                        "statement": "Albert Einstein was a genius.",
                        "reason": "The context and statement are unrelated",
                        "verdict": 0,
                    },
                ]
            ).dicts(),
        },
    ],
    input_keys=["context", "statements"],
    output_key="answer",
    output_type="json",
)  # noqa: E501


@dataclass
class Faithfulness(MetricWithLLM):
    name: str = "faithfulness"  # type: ignore
    evaluation_mode: EvaluationMode = EvaluationMode.qac  # type: ignore
    long_form_answer_prompt: Prompt = field(
        default_factory=lambda: LONG_FORM_ANSWER_PROMPT
    )
    nli_statements_message: Prompt = field(
        default_factory=lambda: NLI_STATEMENTS_MESSAGE
    )
    max_retries: int = 1

    def _create_answer_prompt(self, row: t.Dict) -> PromptValue:
        question, answer = row["question"], row["answer"]

        # extract statements from answer given the question
        prompt_value = self.long_form_answer_prompt.format(
            question=question, answer=answer
        )
        return prompt_value

    def _create_nli_prompt(self, row: t.Dict, statements: t.List[str]) -> PromptValue:
        assert self.llm is not None, "llm must be set to compute score"

        contexts = row["contexts"]
        # check if the statements are support in the contexts
        contexts_str: str = "\n".join(contexts)
        statements_str: str = json.dumps(statements)
        prompt_value = self.nli_statements_message.format(
            context=contexts_str, statements=statements_str
        )
        return prompt_value

    def _compute_score(self, answers: StatementFaithfulnessAnswers):
        # check the verdicts and compute the score
        faithful_statements = sum(
            1 if answer.verdict else 0 for answer in answers.__root__
        )
        num_statements = len(answers.__root__)
        if num_statements:
            score = faithful_statements / num_statements
        else:
            logger.warning("No statements were generated from the answer.")
            score = np.nan

        return score

    async def _ascore(
        self: t.Self, row: t.Dict, callbacks: Callbacks, is_async: bool
    ) -> float:
        """
        returns the NLI score for each (q, c, a) pair
        """
        assert self.llm is not None, "LLM is not set"
        p_value = self._create_answer_prompt(row)
        answer_result = await self.llm.generate(
            p_value, callbacks=callbacks, is_async=is_async
        )
        answer_result_text = answer_result.generations[0][0].text
        statements = await _statements_output_parser.aparse(
            answer_result_text, p_value, self.llm, self.max_retries
        )
        if statements is None:
            return np.nan

        p_value = self._create_nli_prompt(row, statements.__root__)
        nli_result = await self.llm.generate(
            p_value, callbacks=callbacks, is_async=is_async
        )
        nli_result_text = nli_result.generations[0][0].text

        faithfulness = await _faithfulness_output_parser.aparse(
            nli_result_text, p_value, self.llm, self.max_retries
        )
        if faithfulness is None:
            return np.nan

        return self._compute_score(faithfulness)

    def adapt(self, language: str, cache_dir: t.Optional[str] = None) -> None:
        assert self.llm is not None, "LLM is not set"

        logger.info(f"Adapting Faithfulness metric to {language}")
        self.long_form_answer_prompt = self.long_form_answer_prompt.adapt(
            language, self.llm, cache_dir
        )
        self.nli_statements_message = self.nli_statements_message.adapt(
            language, self.llm, cache_dir
        )

    def save(self, cache_dir: t.Optional[str] = None) -> None:
        self.long_form_answer_prompt.save(cache_dir)
        self.nli_statements_message.save(cache_dir)


faithfulness = Faithfulness()

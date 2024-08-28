from __future__ import annotations

import logging
import typing as t
from dataclasses import dataclass, field

import numpy as np
from langchain_core.pydantic_v1 import BaseModel

from ragas.llms.output_parser import RagasoutputParser, get_json_format_instructions
from ragas.llms.prompt import Prompt
from ragas.metrics.base import EvaluationMode, MetricWithEmbeddings, MetricWithLLM

logger = logging.getLogger(__name__)

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks

    from ragas.llms.prompt import PromptValue


class AnswerRelevanceClassification(BaseModel):
    question: str
    noncommittal: int


_output_instructions = get_json_format_instructions(
    pydantic_object=AnswerRelevanceClassification
)
_output_parser = RagasoutputParser(pydantic_object=AnswerRelevanceClassification)


QUESTION_GEN = Prompt(
    name="question_generation",
    instruction="""与えられたanswerに対してquestionを生成し、そのanswerが非断定的であるかを識別してください。answerが非断定的であれば「1」を、断定的であれば「0」を与えてください。非断定的なanswerとは、回避的、曖昧、または曖昧な回答を指します。例えば、「わかりません」や「確かではありません」は非断定的なanswerです。""",
    output_format_instruction=_output_instructions,
    examples=[
        {
            "answer": """アルベルト・アインシュタインはドイツで生まれました。""",
            "context": """アルベルト・アインシュタインはドイツ生まれの理論物理学者で、史上最も偉大で影響力のある科学者の一人と広く見なされています。""",
            "output": AnswerRelevanceClassification.parse_obj(
                {
                    "question": "アルベルト・アインシュタインはどこで生まれましたか？",
                    "noncommittal": 0,
                }
            ).dict(),
        },
        {
            "answer": """それは環境の温度に応じて皮膚の色を変えることができます。""",
            "context": """最近の科学的研究では、アマゾンの熱帯雨林で新種のカエルが発見されました。そのカエルには環境の温度に応じて皮膚の色を変えるユニークな能力があります。""",
            "output": AnswerRelevanceClassification.parse_obj(
                {
                    "question": "新たに発見されたカエルの種にはどのような独自の能力がありますか？",
                    "noncommittal": 0,
                }
            ).dict(),
        },
        {
            "answer": """エベレスト""",
            "context": """海面からの高さで測定した場合、地球で最も高い山はヒマラヤにある有名な峰です。""",
            "output": AnswerRelevanceClassification.parse_obj(
                {
                    "question": "地球で最も高い山は何ですか？",
                    "noncommittal": 0,
                }
            ).dict(),
        },
        {
            "answer": """私は2023年に発明されたスマートフォンの画期的な機能については知りません。2022年以降の情報を持っていないためです。""",
            "context": """2023年に画期的な発明が発表されました。それは1か月間のバッテリー寿命を持つスマートフォンで、モバイル技術の利用方法を一変させました。""",
            "output": AnswerRelevanceClassification.parse_obj(
                {
                    "question": "2023年に発明されたスマートフォンの画期的な機能は何でしたか？",
                    "noncommittal": 1,
                }
            ).dict(),
        },
    ],
    input_keys=["answer", "context"],
    output_key="output",
    output_type="json",
)


@dataclass
class AnswerRelevancy(MetricWithLLM, MetricWithEmbeddings):
    """
    Scores the relevancy of the answer according to the given question.
    Answers with incomplete, redundant or unnecessary information is penalized.
    Score can range from 0 to 1 with 1 being the best.

    Attributes
    ----------
    name: string
        The name of the metrics
    strictness: int
        Here indicates the number questions generated per answer.
        Ideal range between 3 to 5.
    embeddings: Embedding
        The langchain wrapper of Embedding object.
        E.g. HuggingFaceEmbeddings('BAAI/bge-base-en')
    """

    name: str = "answer_relevancy"  # type: ignore
    evaluation_mode: EvaluationMode = EvaluationMode.qac  # type: ignore
    question_generation: Prompt = field(default_factory=lambda: QUESTION_GEN)
    strictness: int = 3

    def calculate_similarity(
        self: t.Self, question: str, generated_questions: list[str]
    ):
        assert self.embeddings is not None
        question_vec = np.asarray(self.embeddings.embed_query(question)).reshape(1, -1)
        gen_question_vec = np.asarray(
            self.embeddings.embed_documents(generated_questions)
        ).reshape(len(generated_questions), -1)
        norm = np.linalg.norm(gen_question_vec, axis=1) * np.linalg.norm(
            question_vec, axis=1
        )
        return (
            np.dot(gen_question_vec, question_vec.T).reshape(
                -1,
            )
            / norm
        )

    def _calculate_score(
        self, answers: t.Sequence[AnswerRelevanceClassification], row: t.Dict
    ) -> float:
        question = row["question"]
        gen_questions = [answer.question for answer in answers]
        committal = np.any([answer.noncommittal for answer in answers])
        if all(q == "" for q in gen_questions):
            logger.warning(
                "Invalid JSON response. Expected dictionary with key 'question'"
            )
            score = np.nan
        else:
            cosine_sim = self.calculate_similarity(question, gen_questions)
            score = cosine_sim.mean() * int(not committal)

        return score

    def _create_question_gen_prompt(self, row: t.Dict) -> PromptValue:
        ans, ctx = row["answer"], row["contexts"]
        return self.question_generation.format(answer=ans, context="\n".join(ctx))

    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        assert self.llm is not None, "LLM is not set"

        prompt = self._create_question_gen_prompt(row)
        result = await self.llm.generate(
            prompt,
            n=self.strictness,
            callbacks=callbacks,
        )

        answers = [
            await _output_parser.aparse(result.text, prompt, self.llm)
            for result in result.generations[0]
        ]
        if any(answer is None for answer in answers):
            return np.nan

        answers = [answer for answer in answers if answer is not None]
        return self._calculate_score(answers, row)

    def adapt(self, language: str, cache_dir: str | None = None) -> None:
        assert self.llm is not None, "LLM is not set"

        logger.info(f"Adapting AnswerRelevancy metric to {language}")
        self.question_generation = self.question_generation.adapt(
            language, self.llm, cache_dir
        )

    def save(self, cache_dir: str | None = None) -> None:
        self.question_generation.save(cache_dir)


answer_relevancy = AnswerRelevancy()

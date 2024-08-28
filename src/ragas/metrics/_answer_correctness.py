from __future__ import annotations

import logging
import typing as t
from dataclasses import dataclass, field

import numpy as np
from langchain_core.pydantic_v1 import BaseModel

from ragas.llms.output_parser import RagasoutputParser, get_json_format_instructions
from ragas.llms.prompt import Prompt, PromptValue
from ragas.metrics._answer_similarity import AnswerSimilarity
from ragas.metrics._faithfulness import (
    LONG_FORM_ANSWER_PROMPT,
    HasSegmentMethod,
    _statements_output_parser,
)
from ragas.metrics.base import (
    EvaluationMode,
    MetricWithEmbeddings,
    MetricWithLLM,
    get_segmenter,
)
from ragas.run_config import RunConfig

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks


logger = logging.getLogger(__name__)


class AnswerCorrectnessClassification(BaseModel):
    TP: t.List[t.Dict[str, t.Any]]
    FP: t.List[t.Dict[str, t.Any]]
    FN: t.List[t.Dict[str, t.Any]]


_output_instructions = get_json_format_instructions(AnswerCorrectnessClassification)
_output_parser = RagasoutputParser(pydantic_object=AnswerCorrectnessClassification)

CORRECTNESS_INSTRUCTIONS = """\
与えられた真の正解（ground truth） と回答文（answer）をもとに、それぞれの文を分析し、次のいずれかのカテゴリーに分類してください:

- TP（真陽性）：回答に含まれており、かつグラウンドトゥルースの1つ以上の文によって直接支持されている文
- FP（偽陽性）：回答に含まれているが、グラウンドトゥルースのいずれの文からも直接支持されていない文
- FN（偽陰性）：グラウンドトゥルースに含まれているが、回答には含まれていない文

各文は1つのカテゴリーにのみ属することができます。それぞれの分類理由を記述してください。
"""
CORRECTNESS_PROMPT = Prompt(
    name="answer_correctness",
    instruction=CORRECTNESS_INSTRUCTIONS,
    output_format_instruction=_output_instructions,
    examples=[
        {
            "question": """太陽のエネルギー源は何で、その主な機能は何ですか？""",
            "answer": [
                "太陽は地球上の原子炉に似た核分裂によってエネルギーを得ています。",
                "太陽の主な機能は、太陽系に光を提供することです。",
            ],
            "ground_truth": [
                "太陽は核融合によってエネルギーを得ており、水素原子が融合してヘリウムを形成します。",
                "太陽の中心部でのこの融合過程は、莫大なエネルギーを放出します。",
                "太陽からのエネルギーは、地球にとって不可欠な熱と光を提供します。",
                "太陽光は地球の気候システムにおいて重要な役割を果たしています。",
                "太陽光は天候や海流を駆動するのを助けます。",
            ],
            "classification": AnswerCorrectnessClassification.parse_obj(
                {
                    "TP": [
                        {
                            "statement": "太陽の主な機能は、太陽系に光を提供することです。",
                            "reason": "この記述は、太陽が光を提供することとその役割に言及している点で、ある程度真実と一致しますが、太陽のエネルギーにより広く焦点を当てています。",
                        }
                    ],
                    "FP": [
                        {
                            "statement": "太陽は地球上の原子炉に似た核分裂によってエネルギーを得ています。",
                            "reason": "この記述は間違っており、太陽が核融合によってエネルギーを得ているとする真実に反しています。",
                        }
                    ],
                    "FN": [
                        {
                            "statement": "太陽は核融合によってエネルギーを得ており、水素原子が融合してヘリウムを形成します。",
                            "reason": "太陽のエネルギー源についてのこの正確な説明は、回答に含まれていません。",
                        },
                        {
                            "statement": "太陽の中心部でのこの融合過程は、莫大なエネルギーを放出します。",
                            "reason": "この過程とその重要性は回答に言及されていません。",
                        },
                        {
                            "statement": "太陽からのエネルギーは、地球にとって不可欠な熱と光を提供します。",
                            "reason": "回答は光についてのみ言及しており、熱とその生命にとっての必要性については省略されています。",
                        },
                        {
                            "statement": "太陽光は地球の気候システムにおいて重要な役割を果たしています。",
                            "reason": "太陽光の地球の気候システムへのこの広い影響は、回答では触れられていません。",
                        },
                        {
                            "statement": "太陽光は天候や海流を駆動するのを助けます。",
                            "reason": "太陽光が天候のパターンや海流に与える影響は、回答で省略されています。",
                        },
                    ],
                }
            ).dict(),
        },
        {
            "question": """水の沸点は何ですか？""",
            "answer": [
                "水の沸点は海抜で100度セルシウスです"
            ],
            "ground_truth": [
                "水の沸点は海抜で100度セルシウス（212度ファーレンハイト）です。",
                "水の沸点は高度によって変わることがあります。",
            ],
            "classification": AnswerCorrectnessClassification.parse_obj(
                {
                    "TP": [
                        {
                            "statement": "水の沸点は海抜で100度セルシウスです",
                            "reason": "この記述は、海抜で水の沸点が100度セルシウスと指定している点で真実に直接一致します。",
                        }
                    ],
                    "FP": [],
                    "FN": [
                        {
                            "statement": "水の沸点は高度によって変わることがあります。",
                            "reason": "水の沸点が高度によって変わるという追加情報は回答に言及されていません。",
                        }
                    ],
                }
            ).dict(),
        },
    ],
    input_keys=["question", "answer", "ground_truth"],
    output_key="classification",
    output_type="json",
)


@dataclass
class AnswerCorrectness(MetricWithLLM, MetricWithEmbeddings):
    """
    Measures answer correctness compared to ground truth as a combination of
    factuality and semantic similarity.

    Attributes
    ----------
    name: string
        The name of the metrics
    weights:
        a list of two weights corresponding to factuality and semantic similarity
        Defaults [0.75, 0.25]
    answer_similarity:
        The AnswerSimilarity object
    """

    name: str = "answer_correctness"  # type: ignore[reportIncompatibleMethodOverride]
    evaluation_mode: EvaluationMode = EvaluationMode.qga  # type: ignore[reportIncompatibleMethodOverride]
    correctness_prompt: Prompt = field(default_factory=lambda: CORRECTNESS_PROMPT)
    long_form_answer_prompt: Prompt = field(
        default_factory=lambda: LONG_FORM_ANSWER_PROMPT
    )
    weights: list[float] = field(default_factory=lambda: [0.75, 0.25])
    answer_similarity: t.Optional[AnswerSimilarity] = None
    sentence_segmenter: t.Optional[HasSegmentMethod] = None
    max_retries: int = 1

    def __post_init__(self: t.Self):
        if len(self.weights) != 2:
            raise ValueError(
                "Expects a list of two weights. First for factuality, second for semantic similarity"
            )
        if all([w == 0 for w in self.weights]):
            raise ValueError("At least one weight must be non-zero")
        if not all([w >= 0 for w in self.weights]):
            raise ValueError("Weights must be non-negative")

        if self.sentence_segmenter is None:
            language = self.long_form_answer_prompt.language
            self.sentence_segmenter = get_segmenter(language=language, clean=False)

    def init(self, run_config: RunConfig):
        super().init(run_config)
        if self.answer_similarity is None and self.weights[1] != 0:
            self.answer_similarity = AnswerSimilarity(
                llm=self.llm, embeddings=self.embeddings
            )

    def _compute_statement_presence(
        self, prediction: AnswerCorrectnessClassification
    ) -> float:
        tp = len(prediction.TP)
        fp = len(prediction.FP)
        fn = len(prediction.FN)
        score = tp / (tp + 0.5 * (fp + fn)) if tp > 0 else 0
        return score

    def _create_statements_prompt(self, question: str, text: str) -> PromptValue:
        assert self.sentence_segmenter is not None, "sentence_segmenter is not set"

        sentences = self.sentence_segmenter.segment(text)
        sentences = [
            sentence for sentence in sentences if sentence.strip().endswith(".")
        ]
        sentences = "\n".join([f"{i}:{x}" for i, x in enumerate(sentences)])
        prompt_value = self.long_form_answer_prompt.format(
            question=question, answer=text, sentences=sentences
        )
        return prompt_value

    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        assert self.llm is not None, "LLM must be set"

        question = row["question"]
        statements = {}
        for item in ["answer", "ground_truth"]:
            p_value = self._create_statements_prompt(question, row[item])
            item_statement = await self.llm.generate(p_value, callbacks=callbacks)
            statements[item] = await _statements_output_parser.aparse(
                item_statement.generations[0][0].text,
                p_value,
                self.llm,
                self.max_retries,
            )
            statements[item] = (
                statements[item].dicts() if statements[item] is not None else []
            )

        if not all([val == [] for val in statements.values()]):
            ground_truth = [
                statement
                for item in statements["ground_truth"]
                for statement in item["simpler_statements"]
            ]
            answer = [
                statement
                for item in statements["answer"]
                for statement in item["simpler_statements"]
            ]
            p_value = self.correctness_prompt.format(
                question=question,
                ground_truth=ground_truth,
                answer=answer,
            )
            is_statement_present = await self.llm.generate(p_value, callbacks=callbacks)
            result_text = is_statement_present.generations[0][0].text

            answers = await _output_parser.aparse(
                result_text, p_value, self.llm, self.max_retries
            )
            if answers is None:
                return np.nan

            f1_score = self._compute_statement_presence(answers)
        else:
            f1_score = 1.0

        if self.weights[1] == 0:
            similarity_score = 0.0
        else:
            assert self.answer_similarity is not None, "AnswerSimilarity must be set"

            similarity_score = await self.answer_similarity.ascore(
                row, callbacks=callbacks
            )

        score = np.average(
            [f1_score, similarity_score],
            weights=self.weights,
        )

        return float(score)

    def adapt(self, language: str, cache_dir: t.Optional[str] = None) -> None:
        assert self.llm is not None, "llm must be set to compute score"

        logger.info(f"Adapting AnswerCorrectness metric to {language}")
        self.correctness_prompt = self.correctness_prompt.adapt(
            language, self.llm, cache_dir
        )

        self.sentence_segmenter = get_segmenter(language=language, clean=False)

    def save(self, cache_dir: t.Optional[str] = None) -> None:
        self.correctness_prompt.save(cache_dir)


answer_correctness = AnswerCorrectness()

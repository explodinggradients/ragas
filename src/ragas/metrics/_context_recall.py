from __future__ import annotations

import logging
import typing as t
from dataclasses import dataclass, field

import numpy as np
from langchain_core.pydantic_v1 import BaseModel

from ragas.llms.output_parser import RagasoutputParser, get_json_format_instructions
from ragas.llms.prompt import Prompt
from ragas.metrics.base import EvaluationMode, MetricWithLLM, ensembler

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
    instruction="""与えられたcontextとanswerに基づき、answer内の各文を分析し、その文が与えられたcontextに帰属できるかどうかを分類してください。分類は"Yes" (1) または "No" (0) の二値分類のみを使用してください。理由とともにJSON形式で出力してください。""",
    output_format_instruction=_classification_output_instructions,
    examples=[
        {
            "question": """アルベルト・アインシュタインについて教えてください。""",
            "context": """アルベルト・アインシュタイン（1879年3月14日 - 1955年4月18日）は、ドイツ生まれの理論物理学者であり、史上最も偉大で影響力のある科学者の一人と広く見なされています。相対性理論の開発で最もよく知られている彼は、量子力学にも重要な貢献をし、20世紀初頭に現代物理学が成し遂げた自然の科学的理解の革命的再編成の中心人物でもありました。彼の質量-エネルギー等価の公式E=mc²は「世界で最も有名な方程式」と呼ばれています。彼は1921年に「理論物理学への貢献と特に光電効果の法則の発見のために」ノーベル物理学賞を受賞しました。彼の業績はまた、科学の哲学に対する影響でも知られています。1999年、イギリスのジャーナル「Physics World」が世界中の130人の著名な物理学者に対して行った世論調査では、アインシュタインは史上最高の物理学者に選ばれました。彼の知的成果と独創性により、アインシュタインは「天才」の代名詞となっています。""",
            "answer": """アルベルト・アインシュタインは1879年3月14日に生まれたドイツ生まれの理論物理学者で、史上最も偉大で影響力のある科学者の一人と広く見なされています。彼は理論物理学への貢献により、1921年にノーベル物理学賞を受賞しました。彼は1905年に4本の論文を発表しました。アインシュタインは1895年にスイスに移住しました。""",
            "classification": ContextRecallClassificationAnswers.parse_obj(
                [
                    {
                        "statement": "アルベルト・アインシュタインは1879年3月14日に生まれたドイツ生まれの理論物理学者で、史上最も偉大で影響力のある科学者の一人と広く見なされています。",
                        "reason": "アインシュタインの生年月日はコンテキストに明確に記載されています。",
                        "attributed": 1,
                    },
                    {
                        "statement": "彼は理論物理学への貢献により、1921年にノーベル物理学賞を受賞しました。",
                        "reason": "この正確な文は与えられたコンテキストに存在します。",
                        "attributed": 1,
                    },
                    {
                        "statement": "彼は1905年に4本の論文を発表しました。",
                        "reason": "彼が書いた論文についての言及はコンテキストにはありません。",
                        "attributed": 0,
                    },
                    {
                        "statement": "アインシュタインは1895年にスイスに移住しました。",
                        "reason": "このことを裏付ける証拠は与えられたコンテキストにはありません。",
                        "attributed": 0,
                    },
                ]
            ).dicts(),
        },
        {
            "question": """2020年のICCワールドカップの優勝者は誰ですか？""",
            "context": """2022年のICC男子T20ワールドカップは、2022年10月16日から11月13日までオーストラリアで開催され、トーナメントの第8回目でした。元々は2020年に予定されていましたが、COVID-19のパンデミックのため延期されました。イングランドが決勝でパキスタンを5ウィケットで破り、2度目のICC男子T20ワールドカップタイトルを獲得しました。""",
            "answer": """イングランド""",
            "classification": ContextRecallClassificationAnswers.parse_obj(
                [
                    {
                        "statement": "イングランドは2022年のICC男子T20ワールドカップで優勝しました。",
                        "reason": "コンテキストから、イングランドがパキスタンを破ってワールドカップで優勝したことが明らかです。",
                        "attributed": 1,
                    },
                ]
            ).dicts(),
        },
        {
            "question": """太陽の主な燃料は何ですか？""",
            "context": """NULL""",
            "answer": """水素""",
            "classification": ContextRecallClassificationAnswers.parse_obj(
                [
                    {
                        "statement": "太陽の主な燃料は水素です。",
                        "reason": "コンテキストには情報が含まれていません。",
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
        qstn, ctx, gt = row["question"], row["contexts"], row["ground_truth"]
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


context_recall = ContextRecall()

from __future__ import annotations

import logging
import typing as t
from dataclasses import dataclass, field

import numpy as np
from langchain.pydantic_v1 import BaseModel, Field

from ragas.llms.output_parser import RagasoutputParser, get_json_format_instructions
from ragas.llms.prompt import Prompt, PromptValue
from ragas.metrics.base import EvaluationMode, MetricWithLLM, ensembler

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks

logger = logging.getLogger(__name__)


class ContextPrecisionVerification(BaseModel):
    """Answer for the verification task wether the context was useful."""

    reason: str = Field(..., description="Reason for verification")
    verdict: int = Field(..., description="Binary (0/1) verdict of verification")


class ContextPrecisionVerifications(BaseModel):
    __root__: t.List[ContextPrecisionVerification]


_verification_output_instructions = get_json_format_instructions(
    ContextPrecisionVerification
)
_output_parser = RagasoutputParser(pydantic_object=ContextPrecisionVerification)

CONTEXT_PRECISION = Prompt(
    name="context_precision",
    instruction="""与えられたquestion、answer、およびcontextを基に、そのcontextが与えられたanswerを導き出すのに役立ったかどうかを検証してください。有用であれば"1"を、そうでなければ"0"を出力し、結果をJSON形式で出力してください。""",
    output_format_instruction=_verification_output_instructions,
    examples=[
        {
            "question": """アルベルト・アインシュタインについて教えてください。""",
            "context": """アルベルト・アインシュタイン（1879年3月14日 - 1955年4月18日）は、ドイツ生まれの理論物理学者であり、史上最も偉大で影響力のある科学者の一人と広く見なされています。彼は相対性理論の開発で最もよく知られていますが、量子力学にも重要な貢献をし、20世紀初頭の現代物理学が達成した自然の科学的理解の革命的再編成の中心人物でもありました。彼の質量とエネルギーの等価性を示す公式 E = mc² は「世界で最も有名な方程式」と呼ばれています。彼は1921年に「理論物理学への貢献と特に光電効果の法則の発見のために」ノーベル物理学賞を受賞しました。彼の仕事はまた、科学の哲学にも影響を与えたことで知られています。1999年にイギリスの雑誌「Physics World」が世界中の130人の著名な物理学者に対して行った世論調査では、アインシュタインは史上最高の物理学者に選ばれました。彼の知的成果と独創性は、アインシュタインを「天才」の代名詞としています。""",
            "answer": """アルベルト・アインシュタインは1879年3月14日に生まれたドイツ生まれの理論物理学者で、史上最も偉大で影響力のある科学者の一人と広く見なされています。彼は理論物理学への貢献により、1921年にノーベル物理学賞を受賞しました。彼は1905年に4本の論文を発表しました。アインシュタインは1895年にスイスに移住しました。""",
            "verification": ContextPrecisionVerification(
                reason="提供されたコンテキストは、与えられた回答を得るために確かに有用でした。コンテキストには、アルベルト・アインシュタインの人生と貢献に関する重要な情報が含まれており、それらが回答に反映されています。",
                verdict=1,
            ).dict(),
        },
        {
            "question": """2020年のICCワールドカップの優勝者は誰ですか？""",
            "context": """2022年のICC男子T20ワールドカップは、2022年10月16日から11月13日までオーストラリアで開催され、トーナメントの第8回目でした。元々は2020年に予定されていましたが、COVID-19のパンデミックのため延期されました。イングランドが決勝でパキスタンを5ウィケットで破り、2度目のICC男子T20ワールドカップタイトルを獲得しました。""",
            "answer": """イングランド""",
            "verification": ContextPrecisionVerification(
                reason="コンテキストは、2020年のICCワールドカップに関する状況を明確にし、2020年に予定されていたトーナメントが実際には2022年に開催され、イングランドが優勝したことを示すのに役立ちました。",
                verdict=1,
            ).dict(),
        },
        {
            "question": """世界で最も高い山は何ですか？""",
            "context": """アンデス山脈は世界で最も長い大陸山脈で、南アメリカに位置しています。それは7つの国にまたがり、西半球で最も高い山々の多くを特徴としています。この山脈は、高地のアンデス高原やアマゾン熱帯雨林を含む多様な生態系で知られています。""",
            "answer": """エベレスト山。""",
            "verification": ContextPrecisionVerification(
                reason="提供されたコンテキストは、アンデス山脈について述べていますが、それは印象的であるものの、エベレスト山や世界で最も高い山に直接関係するものではありません。",
                verdict=0,
            ).dict(),
        },
    ],
    input_keys=["question", "context", "answer"],
    output_key="verification",
    output_type="json",
)


@dataclass
class ContextPrecision(MetricWithLLM):
    """
    Average Precision is a metric that evaluates whether all of the
    relevant items selected by the model are ranked higher or not.

    Attributes
    ----------
    name : str
    evaluation_mode: EvaluationMode
    context_precision_prompt: Prompt
    """

    name: str = "context_precision"  # type: ignore
    evaluation_mode: EvaluationMode = EvaluationMode.qcg  # type: ignore
    context_precision_prompt: Prompt = field(default_factory=lambda: CONTEXT_PRECISION)
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

    def _get_row_attributes(self, row: t.Dict) -> t.Tuple[str, t.List[str], t.Any]:
        return row["question"], row["contexts"], row["ground_truth"]

    def _context_precision_prompt(self, row: t.Dict) -> t.List[PromptValue]:
        question, contexts, answer = self._get_row_attributes(row)
        return [
            self.context_precision_prompt.format(
                question=question, context=c, answer=answer
            )
            for c in contexts
        ]

    def _calculate_average_precision(
        self, verifications: t.List[ContextPrecisionVerification]
    ) -> float:
        score = np.nan

        verdict_list = [1 if ver.verdict else 0 for ver in verifications]
        denominator = sum(verdict_list) + 1e-10
        numerator = sum(
            [
                (sum(verdict_list[: i + 1]) / (i + 1)) * verdict_list[i]
                for i in range(len(verdict_list))
            ]
        )
        score = numerator / denominator
        if np.isnan(score):
            logger.warning(
                "Invalid response format. Expected a list of dictionaries with keys 'verdict'"
            )
        return score

    async def _ascore(
        self: t.Self,
        row: t.Dict,
        callbacks: Callbacks,
    ) -> float:
        assert self.llm is not None, "LLM is not set"

        human_prompts = self._context_precision_prompt(row)
        responses = []
        for hp in human_prompts:
            results = await self.llm.generate(
                hp,
                callbacks=callbacks,
                n=self.reproducibility,
            )
            results = [
                await _output_parser.aparse(item.text, hp, self.llm, self.max_retries)
                for item in results.generations[0]
            ]

            responses.append(
                [result.dict() for result in results if result is not None]
            )

        answers = []
        for response in responses:
            agg_answer = ensembler.from_discrete([response], "verdict")
            if agg_answer:
                agg_answer = ContextPrecisionVerification.parse_obj(agg_answer[0])
            answers.append(agg_answer)

        answers = ContextPrecisionVerifications(__root__=answers)
        score = self._calculate_average_precision(answers.__root__)
        return score

    def adapt(self, language: str, cache_dir: str | None = None) -> None:
        assert self.llm is not None, "LLM is not set"

        logging.info(f"Adapting Context Precision to {language}")
        self.context_precision_prompt = self.context_precision_prompt.adapt(
            language, self.llm, cache_dir
        )

    def save(self, cache_dir: str | None = None) -> None:
        self.context_precision_prompt.save(cache_dir)


@dataclass
class ContextUtilization(ContextPrecision):
    name: str = "context_utilization"
    evaluation_mode: EvaluationMode = EvaluationMode.qac

    def _get_row_attributes(self, row: t.Dict) -> t.Tuple[str, t.List[str], t.Any]:
        return row["question"], row["contexts"], row["answer"]


context_precision = ContextPrecision()
context_utilization = ContextUtilization()

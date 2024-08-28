from __future__ import annotations

import logging
import typing as t
from dataclasses import dataclass, field
from typing import Dict

import numpy as np
from langchain.pydantic_v1 import BaseModel

from ragas.llms.output_parser import RagasoutputParser, get_json_format_instructions
from ragas.llms.prompt import Prompt
from ragas.metrics.base import EvaluationMode, MetricWithLLM

if t.TYPE_CHECKING:
    from langchain.callbacks.base import Callbacks

logger = logging.getLogger(__name__)


class ContextEntitiesResponse(BaseModel):
    entities: t.List[str]


_output_instructions = get_json_format_instructions(
    pydantic_object=ContextEntitiesResponse
)
_output_parser = RagasoutputParser(pydantic_object=ContextEntitiesResponse)


TEXT_ENTITY_EXTRACTION = Prompt(
    name="text_entity_extraction",
    instruction="""与えられたテキストから、重複なくユニークなエンティティを抽出してください。同じエンティティの異なる形や言及も、単一のエンティティとして扱うようにしてください。""",
    input_keys=["text"],
    output_key="output",
    output_type="json",
    output_format_instruction=_output_instructions,
    examples=[
        {
            "text": """エッフェル塔は、フランスのパリに位置する世界的に最も象徴的なランドマークの一つです。
            毎年何百万人もの訪問者が、その美しい市内の眺望を求めて訪れます。
            1889年に完成し、1889年の万国博覧会に間に合うように建設されました。""",
            "output": ContextEntitiesResponse.parse_obj(
                {
                    "entities": [
                        "エッフェル塔",
                        "パリ",
                        "フランス",
                        "1889年",
                        "万国博覧会",
                    ],
                }
            ).dict(),
        },
        {
            "text": """ローマのコロッセオは、フラヴィウス円形劇場としても知られており、ローマの建築と工学の偉業を示す記念碑です。
            建設は紀元70年に皇帝ウェスパシアヌスの下で始まり、息子ティトゥスによって紀元80年に完成しました。
            5万人から8万人の観客を収容でき、彼らは剣闘士の競技や公開スペクタクルを観戦しました。""",
            "output": ContextEntitiesResponse.parse_obj(
                {
                    "entities": [
                        "コロッセオ",
                        "ローマ",
                        "フラヴィウス円形劇場",
                        "ウェスパシアヌス",
                        "紀元70年",
                        "ティトゥス",
                        "紀元80年",
                    ],
                }
            ).dict(),
        },
        {
            "text": """万里の長城は東から西へ21,196キロメートル以上にわたって伸びており、古代の防御建築の驚異です。
            北からの侵略に対する防御として建てられ、その建設は紀元前7世紀に始まりました。
            現在では、ユネスコ世界遺産に登録されており、大きな観光名所となっています。""",
            "output": ContextEntitiesResponse.parse_obj(
                {
                    "entities": [
                        "万里の長城",
                        "21,196キロメートル",
                        "紀元前7世紀",
                        "ユネスコ世界遺産",
                    ],
                }
            ).dict(),
        },
        {
            "text": """アポロ11号のミッションは、1969年7月16日に打ち上げられ、人類が初めて月に着陸したことを記念しました。
            宇宙飛行士ニール・アームストロング、バズ・オルドリン、マイケル・コリンズが歴史を作り、アームストロングは月面に最初に足を踏み入れた人間となりました。
            この出来事は、宇宙探査における重要なマイルストーンでした。""",
            "output": ContextEntitiesResponse.parse_obj(
                {
                    "entities": [
                        "アポロ11号ミッション",
                        "1969年7月16日",
                        "月",
                        "ニール・アームストロング",
                        "バズ・オルドリン",
                        "マイケル・コリンズ",
                    ],
                }
            ).dict(),
        },
    ],
)


@dataclass
class ContextEntityRecall(MetricWithLLM):
    """
    Calculates recall based on entities present in ground truth and context.
    Let CN be the set of entities present in context,
    GN be the set of entities present in the ground truth.

    Then we define can the context entity recall as follows:
    Context Entity recall = | CN ∩ GN | / | GN |

    If this quantity is 1, we can say that the retrieval mechanism has
    retrieved context which covers all entities present in the ground truth,
    thus being a useful retrieval. Thus this can be used to evaluate retrieval
    mechanisms in specific use cases where entities matter, for example, a
    tourism help chatbot.

    Attributes
    ----------
    name : str
    batch_size : int
        Batch size for openai completion.
    """

    name: str = "context_entity_recall"  # type: ignore
    evaluation_mode: EvaluationMode = EvaluationMode.gc  # type: ignore
    context_entity_recall_prompt: Prompt = field(
        default_factory=lambda: TEXT_ENTITY_EXTRACTION
    )
    batch_size: int = 15
    max_retries: int = 1

    def _compute_score(
        self, ground_truth_entities: t.Sequence[str], context_entities: t.Sequence[str]
    ) -> float:
        num_entities_in_both = len(
            set(context_entities).intersection(set(ground_truth_entities))
        )
        return num_entities_in_both / (len(ground_truth_entities) + 1e-8)

    async def get_entities(
        self,
        text: str,
        callbacks: Callbacks,
    ) -> t.Optional[ContextEntitiesResponse]:
        assert self.llm is not None, "LLM is not initialized"
        p_value = self.context_entity_recall_prompt.format(
            text=text,
        )
        result = await self.llm.generate(
            prompt=p_value,
            callbacks=callbacks,
        )

        result_text = result.generations[0][0].text
        answer = await _output_parser.aparse(
            result_text, p_value, self.llm, self.max_retries
        )
        if answer is None:
            return ContextEntitiesResponse(entities=[])

        return answer

    async def _ascore(
        self,
        row: Dict,
        callbacks: Callbacks,
    ) -> float:
        ground_truth, contexts = row["ground_truth"], row["contexts"]
        ground_truth = await self.get_entities(ground_truth, callbacks=callbacks)
        contexts = await self.get_entities("\n".join(contexts), callbacks=callbacks)
        if ground_truth is None or contexts is None:
            return np.nan
        return self._compute_score(ground_truth.entities, contexts.entities)

    def save(self, cache_dir: str | None = None) -> None:
        return self.context_entity_recall_prompt.save(cache_dir)


context_entity_recall = ContextEntityRecall(batch_size=15)

from __future__ import annotations

import logging
import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass

from pydantic import BaseModel, Field

from ragas.experimental.llms.prompt import PydanticPrompt
from ragas.llms.base import BaseRagasLLM

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks

logger = logging.getLogger(__name__)


class StatementFaithfulnessAnswer(BaseModel):
    statement: str = Field(..., description="the original statement, word-by-word")
    reason: str = Field(..., description="the reason of the verdict")
    verdict: int = Field(..., description="the verdict(0/1) of the faithfulness.")


class NLIStatementOutput(BaseModel):
    statements: t.List[StatementFaithfulnessAnswer]


class NLIStatementInput(BaseModel):
    context: str = Field(..., description="The context of the question")
    statements: t.List[str] = Field(..., description="The statements to judge")


class NLIStatementPrompt(PydanticPrompt[NLIStatementInput, NLIStatementOutput]):
    instruction = "Your task is to judge the faithfulness of a series of statements based on a given context. For each statement you must return verdict as 1 if the statement can be directly inferred based on the context or 0 if the statement can not be directly inferred based on the context."
    input_model = NLIStatementInput
    output_model = NLIStatementOutput
    examples = [
        (
            NLIStatementInput(
                context="""John is a student at XYZ University. He is pursuing a degree in Computer Science. He is enrolled in several courses this semester, including Data Structures, Algorithms, and Database Management. John is a diligent student and spends a significant amount of time studying and completing assignments. He often stays late in the library to work on his projects.""",
                statements=[
                    "John is majoring in Biology.",
                    "John is taking a course on Artificial Intelligence.",
                    "John is a dedicated student.",
                    "John has a part-time job.",
                ],
            ),
            NLIStatementOutput(
                statements=[
                    StatementFaithfulnessAnswer(
                        statement="John is majoring in Biology.",
                        reason="John's major is explicitly mentioned as Computer Science. There is no information suggesting he is majoring in Biology.",
                        verdict=0,
                    ),
                    StatementFaithfulnessAnswer(
                        statement="John is taking a course on Artificial Intelligence.",
                        reason="The context mentions the courses John is currently enrolled in, and Artificial Intelligence is not mentioned. Therefore, it cannot be deduced that John is taking a course on AI.",
                        verdict=0,
                    ),
                    StatementFaithfulnessAnswer(
                        statement="John is a dedicated student.",
                        reason="The context states that he spends a significant amount of time studying and completing assignments. Additionally, it mentions that he often stays late in the library to work on his projects, which implies dedication.",
                        verdict=1,
                    ),
                    StatementFaithfulnessAnswer(
                        statement="John has a part-time job.",
                        reason="There is no information given in the context about John having a part-time job.",
                        verdict=0,
                    ),
                ]
            ),
        ),
        (
            NLIStatementInput(
                context="Photosynthesis is a process used by plants, algae, and certain bacteria to convert light energy into chemical energy.",
                statements=[
                    "Albert Einstein was a genius.",
                ],
            ),
            NLIStatementOutput(
                statements=[
                    StatementFaithfulnessAnswer(
                        statement="Albert Einstein was a genius.",
                        reason="The context and statement are unrelated",
                        verdict=0,
                    )
                ]
            ),
        ),
    ]


class BaseNLIComponent(ABC):
    @abstractmethod
    async def apply(
        self, hypothesis: str, premises: t.List[str], callbacks: Callbacks
    ) -> t.List[bool]:
        """
        Apply the NLI component to a list of premises and a hypothesis.
        """
        raise NotImplementedError("apply method must be implemented by subclasses")


@dataclass
class LLMNLIComponent(BaseNLIComponent):
    llm: BaseRagasLLM
    nli_prompt: PydanticPrompt = NLIStatementPrompt()

    async def apply(
        self, hypothesis: str, premises: t.List[str], callbacks: Callbacks
    ) -> t.List[bool]:
        assert self.llm is not None, "LLM must be set"
        prompt_input = NLIStatementInput(context=hypothesis, statements=premises)
        response = await self.nli_prompt.generate(
            data=prompt_input, llm=self.llm, callbacks=callbacks
        )
        return [bool(result.verdict) for result in response.statements]


@dataclass
class SequenceClassificationNLIComponent(BaseNLIComponent):
    pretrained_model_name_or_path: str = "vectara/hallucination_evaluation_model"
    batch_size: int = 32
    device: str = "cpu"

    def __post_init__(self):
        try:
            from transformers import AutoModelForSequenceClassification
        except ImportError:
            raise ImportError(
                "Huggingface transformers must be installed to use this feature, try `pip install transformers`"
            )
        except Exception as e:
            raise RuntimeError("Failed to load the model") from e
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.pretrained_model_name_or_path, trust_remote_code=True
        )
        self.model.to(self.device)

    def _create_batch(
        self, pairs: t.List[t.Tuple[str, str]]
    ) -> t.Generator[t.List[t.Tuple[str, str]], None, None]:
        length_of_pairs = len(pairs)
        for ndx in range(0, length_of_pairs, self.batch_size):
            yield pairs[ndx : min(ndx + self.batch_size, length_of_pairs)]

    async def apply(
        self, hypothesis: str, premises: t.List[str], callbacks: Callbacks
    ) -> t.List[bool]:
        scores = []
        pairs = [(hypothesis, premise) for premise in premises]
        batch_pairs = self._create_batch(pairs)
        for input_pairs in batch_pairs:  # to avoid OOM
            batch_scores = self.model.predict(input_pairs).cpu().detach().round()
            scores += batch_scores

        return [bool(score) for score in scores]

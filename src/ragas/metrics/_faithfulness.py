from __future__ import annotations

import logging
import typing as t
from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel, Field

from ragas.dataset_schema import MultiTurnSample, SingleTurnSample
from ragas.metrics.base import (
    MetricOutputType,
    MetricType,
    MetricWithLLM,
    SingleTurnMetric,
)
from ragas.prompt import PydanticPrompt

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks
    from langchain_core.prompt_values import PromptValue

    from ragas.llms.batch_api import BatchJob, BatchResponse

logger = logging.getLogger(__name__)


class StatementGeneratorInput(BaseModel):
    question: str = Field(description="The question to answer")
    answer: str = Field(description="The answer to the question")


class StatementGeneratorOutput(BaseModel):
    statements: t.List[str] = Field(description="The generated statements")


class StatementGeneratorPrompt(
    PydanticPrompt[StatementGeneratorInput, StatementGeneratorOutput]
):
    instruction = "Given a question and an answer, analyze the complexity of each sentence in the answer. Break down each sentence into one or more fully understandable statements. Ensure that no pronouns are used in any statement. Format the outputs in JSON."
    input_model = StatementGeneratorInput
    output_model = StatementGeneratorOutput
    examples = [
        (
            StatementGeneratorInput(
                question="Who was Albert Einstein and what is he best known for?",
                answer="He was a German-born theoretical physicist, widely acknowledged to be one of the greatest and most influential physicists of all time. He was best known for developing the theory of relativity, he also made important contributions to the development of the theory of quantum mechanics.",
            ),
            StatementGeneratorOutput(
                statements=[
                    "Albert Einstein was a German-born theoretical physicist.",
                    "Albert Einstein is recognized as one of the greatest and most influential physicists of all time.",
                    "Albert Einstein was best known for developing the theory of relativity.",
                    "Albert Einstein also made important contributions to the development of the theory of quantum mechanics.",
                ]
            ),
        )
    ]


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


@dataclass
class Faithfulness(MetricWithLLM, SingleTurnMetric):
    name: str = "faithfulness"
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {
                "user_input",
                "response",
                "retrieved_contexts",
            }
        }
    )
    output_type: t.Optional[MetricOutputType] = MetricOutputType.CONTINUOUS
    nli_statements_prompt: PydanticPrompt = field(default_factory=NLIStatementPrompt)
    statement_generator_prompt: PydanticPrompt = field(
        default_factory=StatementGeneratorPrompt
    )
    max_retries: int = 1

    async def _create_verdicts(
        self, row: t.Dict, statements: t.List[str], callbacks: Callbacks
    ) -> NLIStatementOutput:
        assert self.llm is not None, "llm must be set to compute score"

        contexts_str: str = "\n".join(row["retrieved_contexts"])
        verdicts = await self.nli_statements_prompt.generate(
            data=NLIStatementInput(context=contexts_str, statements=statements),
            llm=self.llm,
            callbacks=callbacks,
        )

        return verdicts

    async def _create_statements(
        self, row: t.Dict, callbacks: Callbacks
    ) -> StatementGeneratorOutput:
        assert self.llm is not None, "llm is not set"

        text, question = row["response"], row["user_input"]

        prompt_input = StatementGeneratorInput(question=question, answer=text)
        statements = await self.statement_generator_prompt.generate(
            llm=self.llm,
            data=prompt_input,
            callbacks=callbacks,
        )

        return statements

    def _compute_score(self, answers: NLIStatementOutput):
        # check the verdicts and compute the score
        faithful_statements = sum(
            1 if answer.verdict else 0 for answer in answers.statements
        )
        num_statements = len(answers.statements)
        if num_statements:
            score = faithful_statements / num_statements
        else:
            logger.warning("No statements were generated from the answer.")
            score = np.nan

        return score

    def _samples_to_prompts(
        self, samples: t.List[t.Union[SingleTurnSample, MultiTurnSample]]
    ) -> t.List["PromptValue"]:
        """
        Convert samples to PromptValue objects for batch processing.

        For Faithfulness metric, this implementation focuses on the statement generation step
        as it's the most computationally expensive part. The NLI verification step would
        need the statements generated from this batch job and would be handled in a
        separate batch job or through regular evaluation.

        Note: This handles only the statement generation phase of faithfulness evaluation.
        The complete faithfulness score requires a two-step process where NLI verification
        follows statement generation.
        """
        prompts = []
        for i, sample in enumerate(samples):
            if not isinstance(sample, SingleTurnSample):
                raise ValueError(
                    "Faithfulness metric only supports single-turn samples"
                )

            # Convert sample to dict for processing
            sample_dict = sample.model_dump()

            # Create statement generation prompt using the actual PydanticPrompt
            statement_prompt_input = StatementGeneratorInput(
                question=sample_dict["user_input"], answer=sample_dict["response"]
            )

            # Convert the PydanticPrompt to PromptValue
            # For now, we use a simplified conversion - in a real implementation
            # this would use the actual prompt conversion method from PydanticPrompt
            from langchain_core.prompts import ChatPromptTemplate

            # Create a simple prompt template as fallback
            simple_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "user",
                        f"Question: {statement_prompt_input.question}\nAnswer: {statement_prompt_input.answer}\n\nGenerate individual factual statements from this answer.",
                    )
                ]
            )
            prompt_value = simple_prompt.format_prompt()

            prompts.append(prompt_value)

        return prompts

    def create_complete_batch_evaluation_job(
        self,
        samples: t.List[t.Union[SingleTurnSample, MultiTurnSample]],
        batch_size: t.Optional[int] = None,
        metadata: t.Optional[t.Dict[str, str]] = None,
    ) -> "CompleteFaithfulnessBatchJob":
        """
        Create a complete batch evaluation job that handles both statement generation
        and NLI verification steps for faithfulness evaluation.

        This method returns a specialized batch job that orchestrates the two-step
        faithfulness evaluation process in batch mode.
        """
        if not self.supports_batch_evaluation():
            raise ValueError(
                f"Metric '{self.name}' does not support batch evaluation. "
                "Ensure the LLM supports batch API operations."
            )

        if batch_size is None:
            batch_size = 1000

        if len(samples) > batch_size:
            raise ValueError(
                f"Sample count {len(samples)} exceeds maximum batch size {batch_size}. "
                "Consider splitting into smaller batches."
            )

        return CompleteFaithfulnessBatchJob(
            faithfulness_metric=self,
            samples=samples,
            batch_size=batch_size,
            metadata=metadata or {},
        )

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        row = sample.to_dict()
        return await self._ascore(row, callbacks)

    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        """
        returns the NLI score for each (q, c, a) pair
        """
        assert self.llm is not None, "LLM is not set"

        statements = await self._create_statements(row, callbacks)
        statements = statements.statements
        if statements == []:
            return np.nan

        verdicts = await self._create_verdicts(row, statements, callbacks)
        return self._compute_score(verdicts)


@dataclass
class FaithfulnesswithHHEM(Faithfulness):
    name: str = "faithfulness_with_hhem"
    device: str = "cpu"
    batch_size: int = 10

    def __post_init__(self):
        try:
            from transformers import AutoModelForSequenceClassification  # type: ignore
        except ImportError:
            raise ImportError(
                "Huggingface transformers must be installed to use this feature, try `pip install transformers`"
            )
        self.nli_classifier = AutoModelForSequenceClassification.from_pretrained(
            "vectara/hallucination_evaluation_model", trust_remote_code=True
        )
        self.nli_classifier.to(self.device)
        super().__post_init__()

    def _create_pairs(
        self, row: t.Dict, statements: t.List[str]
    ) -> t.List[t.Tuple[str, str]]:
        """
        create pairs of (question, answer) from the row
        """
        premise = "\n".join(row["retrieved_contexts"])
        pairs = [(premise, statement) for statement in statements]
        return pairs

    def _create_batch(
        self, pairs: t.List[t.Tuple[str, str]]
    ) -> t.Generator[t.List[t.Tuple[str, str]], None, None]:
        length_of_pairs = len(pairs)
        for ndx in range(0, length_of_pairs, self.batch_size):
            yield pairs[ndx : min(ndx + self.batch_size, length_of_pairs)]

    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        """
        returns the NLI score for each (q, c, a) pair
        """
        assert self.llm is not None, "LLM is not set"

        statements = await self._create_statements(row, callbacks)
        statements = statements.statements
        if statements == []:
            return np.nan

        scores = []
        pairs = self._create_pairs(row, statements)
        for input_pairs in self._create_batch(pairs):  # to avoid OOM
            batch_scores = (
                self.nli_classifier.predict(input_pairs).cpu().detach().round()
            )
            # convert tensor to list of floats
            scores.extend(batch_scores.tolist())

        return sum(scores) / len(scores)


class CompleteFaithfulnessBatchJob:
    """
    A specialized batch job for complete faithfulness evaluation that handles
    both statement generation and NLI verification in sequence.
    """

    def __init__(
        self,
        faithfulness_metric: Faithfulness,
        samples: t.List[t.Union[SingleTurnSample, MultiTurnSample]],
        batch_size: int = 1000,
        metadata: t.Optional[t.Dict[str, str]] = None,
    ):
        self.faithfulness_metric = faithfulness_metric
        self.samples = samples
        self.batch_size = batch_size
        self.metadata = metadata or {}
        self.statement_job: t.Optional["BatchJob"] = None
        self.nli_job: t.Optional["BatchJob"] = None

    def execute(self) -> t.List[float]:
        """
        Execute the complete faithfulness evaluation in batch mode.

        Returns:
            List of faithfulness scores for each sample
        """
        # Step 1: Generate statements using batch API
        statement_prompts = self.faithfulness_metric._samples_to_prompts(self.samples)

        if self.faithfulness_metric.llm is None:
            raise ValueError("Faithfulness metric has no LLM configured")

        self.statement_job = self.faithfulness_metric.llm.create_batch_job(
            prompts=statement_prompts,
            metadata={
                **self.metadata,
                "step": "statement_generation",
                "metric": "faithfulness",
            },
        )

        # Wait for statement generation to complete
        status = self.statement_job.wait_for_completion()
        if status.value != "completed":
            raise RuntimeError(f"Statement generation batch job failed: {status.value}")

        # Get statement generation results
        statement_responses = self.statement_job.get_results()

        # Parse statements from responses
        all_statements = self._parse_statement_responses(statement_responses)

        # Step 2: Create NLI verification prompts
        nli_prompts = self._create_nli_prompts(all_statements)

        if nli_prompts:  # Only create NLI job if there are statements to verify
            self.nli_job = self.faithfulness_metric.llm.create_batch_job(
                prompts=nli_prompts,
                metadata={
                    **self.metadata,
                    "step": "nli_verification",
                    "metric": "faithfulness",
                },
            )

            # Wait for NLI verification to complete
            nli_status = self.nli_job.wait_for_completion()
            if nli_status.value != "completed":
                raise RuntimeError(
                    f"NLI verification batch job failed: {nli_status.value}"
                )

            # Get NLI results and compute final scores
            nli_responses = self.nli_job.get_results()
            return self._compute_final_scores(all_statements, nli_responses)
        else:
            # No statements were generated, return NaN scores
            return [np.nan] * len(self.samples)

    async def aexecute(self) -> t.List[float]:
        """Async version of execute."""
        # Step 1: Generate statements using batch API
        statement_prompts = self.faithfulness_metric._samples_to_prompts(self.samples)

        if self.faithfulness_metric.llm is None:
            raise ValueError("Faithfulness metric has no LLM configured")

        self.statement_job = await self.faithfulness_metric.llm.acreate_batch_job(
            prompts=statement_prompts,
            metadata={
                **self.metadata,
                "step": "statement_generation",
                "metric": "faithfulness",
            },
        )

        # Wait for statement generation to complete
        status = await self.statement_job.await_completion()
        if status.value != "completed":
            raise RuntimeError(f"Statement generation batch job failed: {status.value}")

        # Get statement generation results
        statement_responses = await self.statement_job.aget_results()

        # Parse statements from responses
        all_statements = self._parse_statement_responses(statement_responses)

        # Step 2: Create NLI verification prompts
        nli_prompts = self._create_nli_prompts(all_statements)

        if nli_prompts:  # Only create NLI job if there are statements to verify
            self.nli_job = await self.faithfulness_metric.llm.acreate_batch_job(
                prompts=nli_prompts,
                metadata={
                    **self.metadata,
                    "step": "nli_verification",
                    "metric": "faithfulness",
                },
            )

            # Wait for NLI verification to complete
            nli_status = await self.nli_job.await_completion()
            if nli_status.value != "completed":
                raise RuntimeError(
                    f"NLI verification batch job failed: {nli_status.value}"
                )

            # Get NLI results and compute final scores
            nli_responses = await self.nli_job.aget_results()
            return self._compute_final_scores(all_statements, nli_responses)
        else:
            # No statements were generated, return NaN scores
            return [np.nan] * len(self.samples)

    def _parse_statement_responses(
        self, responses: t.List["BatchResponse"]
    ) -> t.List[t.List[str]]:
        """Parse statement generation responses to extract statements for each sample."""
        all_statements = []

        for i, response in enumerate(responses):
            statements = []

            if response.error is None and response.response is not None:
                try:
                    # Extract content from OpenAI response
                    content = self._extract_response_content(response.response)

                    # Try to parse as StatementGeneratorOutput
                    statements = self._parse_statement_content(content)

                except Exception as e:
                    logger.warning(f"Failed to parse statements for sample {i}: {e}")

            all_statements.append(statements)

        return all_statements

    def _extract_response_content(self, response: t.Dict[str, t.Any]) -> str:
        """Extract text content from OpenAI response."""
        try:
            choices = response.get("choices", [])
            if choices and len(choices) > 0:
                message = choices[0].get("message", {})
                return message.get("content", "")
            return ""
        except Exception:
            return ""

    def _parse_statement_content(self, content: str) -> t.List[str]:
        """Parse statement generation content to extract individual statements."""
        try:
            import json
            import re

            # Clean content and try to parse as JSON
            content = content.strip()

            # Look for JSON blocks
            json_match = re.search(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL)
            if json_match:
                content = json_match.group(1)
            elif content.startswith("{") and content.endswith("}"):
                pass  # Already clean JSON
            else:
                # Look for JSON object in text
                json_match = re.search(r"\{[^{}]*\}", content)
                if json_match:
                    content = json_match.group(0)
                else:
                    return []

            parsed = json.loads(content)

            # Extract statements from the parsed response
            if "statements" in parsed and isinstance(parsed["statements"], list):
                return [stmt for stmt in parsed["statements"] if isinstance(stmt, str)]

            return []

        except (json.JSONDecodeError, Exception):
            # Fallback: try to extract statements from text
            lines = content.split("\n")
            statements = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith("#") and not line.startswith("*"):
                    # Remove common prefixes like "1.", "-", etc.
                    line = re.sub(r"^\d+\.\s*", "", line)
                    line = re.sub(r"^[-*]\s*", "", line)
                    if line:
                        statements.append(line)
            return statements

    def _create_nli_prompts(
        self, all_statements: t.List[t.List[str]]
    ) -> t.List["PromptValue"]:
        """Create NLI verification prompts for all statements."""
        prompts = []

        for i, (sample, statements) in enumerate(zip(self.samples, all_statements)):
            if not statements:  # Skip if no statements were generated
                continue

            if not isinstance(sample, SingleTurnSample):
                continue  # Skip non-single-turn samples

            # Get context for this sample
            contexts = sample.retrieved_contexts or []
            contexts_str = "\n".join(contexts)

            # Create NLI prompt input
            nli_input = NLIStatementInput(context=contexts_str, statements=statements)

            # Convert to prompt value - simplified implementation
            from langchain_core.prompts import ChatPromptTemplate

            nli_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "user",
                        f"Context: {nli_input.context}\n\nStatements to verify:\n"
                        + "\n".join([f"- {stmt}" for stmt in nli_input.statements])
                        + "\n\nFor each statement, determine if it can be directly inferred from the context. Return 1 for true, 0 for false.",
                    )
                ]
            )
            prompt_value = nli_prompt.format_prompt()

            prompts.append(prompt_value)

        return prompts

    def _compute_final_scores(
        self,
        all_statements: t.List[t.List[str]],
        nli_responses: t.List["BatchResponse"],
    ) -> t.List[float]:
        """Compute final faithfulness scores from NLI responses."""
        scores = []
        response_idx = 0

        for statements in all_statements:
            if not statements:
                # No statements generated for this sample
                scores.append(np.nan)
                continue

            if response_idx >= len(nli_responses):
                # No response available
                scores.append(np.nan)
                continue

            response = nli_responses[response_idx]
            response_idx += 1

            if response.error is not None or response.response is None:
                scores.append(np.nan)
                continue

            try:
                # Parse NLI response
                content = self._extract_response_content(response.response)
                verdicts = self._parse_nli_content(content)

                if verdicts:
                    # Compute faithfulness score as average of verdicts
                    score = sum(verdicts) / len(verdicts)
                    scores.append(score)
                else:
                    scores.append(np.nan)

            except Exception as e:
                logger.warning(f"Failed to parse NLI response: {e}")
                scores.append(np.nan)

        return scores

    def _parse_nli_content(self, content: str) -> t.List[float]:
        """Parse NLI verification content to extract verdict scores."""
        try:
            import json
            import re

            # Clean content and try to parse as JSON
            content = content.strip()

            # Look for JSON blocks
            json_match = re.search(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL)
            if json_match:
                content = json_match.group(1)
            elif content.startswith("{") and content.endswith("}"):
                pass  # Already clean JSON
            else:
                # Look for JSON object in text
                json_match = re.search(r"\{[^{}]*\}", content)
                if json_match:
                    content = json_match.group(0)
                else:
                    return []

            parsed = json.loads(content)

            # Extract verdicts from statements
            if "statements" in parsed and isinstance(parsed["statements"], list):
                verdicts = []
                for stmt in parsed["statements"]:
                    if isinstance(stmt, dict) and "verdict" in stmt:
                        verdict = stmt["verdict"]
                        if isinstance(verdict, (int, float)):
                            verdicts.append(float(verdict))
                return verdicts

            return []

        except (json.JSONDecodeError, Exception):
            return []


faithfulness = Faithfulness()

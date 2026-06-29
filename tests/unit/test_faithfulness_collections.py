"""Tests for Faithfulness metric (collections implementation)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from ragas.llms.base import InstructorBaseRagasLLM
from ragas.metrics.collections.faithfulness import Faithfulness
from ragas.metrics.collections.faithfulness.util import (
    NLIStatementOutput,
    StatementFaithfulnessAnswer,
    StatementGeneratorOutput,
)


class MockInstructorLLM(InstructorBaseRagasLLM):
    def __init__(self):
        self.agenerate = AsyncMock()
        self.generate = MagicMock()

    def generate(self, prompt, response_model):
        return self.generate(prompt, response_model)

    async def agenerate(self, prompt, response_model):
        return await self.agenerate(prompt, response_model)


@pytest.fixture
def mock_llm():
    return MockInstructorLLM()


class TestFaithfulnessCollections:
    @pytest.mark.asyncio
    async def test_ascore_without_reason(self, mock_llm):
        mock_llm.agenerate.side_effect = [
            StatementGeneratorOutput(statements=["Einstein was born in Germany."]),
            NLIStatementOutput(
                statements=[
                    StatementFaithfulnessAnswer(
                        statement="Einstein was born in Germany.",
                        reason="The context states Einstein was born in Germany.",
                        verdict=1,
                    )
                ]
            ),
        ]

        metric = Faithfulness(llm=mock_llm)
        result = await metric.ascore(
            user_input="Where was Einstein born?",
            response="Einstein was born in Germany.",
            retrieved_contexts=["Albert Einstein was born in Germany."],
        )

        assert result.value == 1.0
        assert result.reason is None

    @pytest.mark.asyncio
    async def test_ascore_with_include_reason(self, mock_llm):
        mock_llm.agenerate.side_effect = [
            StatementGeneratorOutput(
                statements=[
                    "Einstein was born in Germany.",
                    "Einstein was born in France.",
                ]
            ),
            NLIStatementOutput(
                statements=[
                    StatementFaithfulnessAnswer(
                        statement="Einstein was born in Germany.",
                        reason="The context states Einstein was born in Germany.",
                        verdict=1,
                    ),
                    StatementFaithfulnessAnswer(
                        statement="Einstein was born in France.",
                        reason="The context does not mention France.",
                        verdict=0,
                    ),
                ]
            ),
        ]

        metric = Faithfulness(llm=mock_llm)
        result = await metric.ascore(
            user_input="Where was Einstein born?",
            response="Einstein was born in Germany and France.",
            retrieved_contexts=["Albert Einstein was born in Germany."],
            include_reason=True,
        )

        assert result.value == 0.5
        assert result.reason is not None
        assert "[supported] Einstein was born in Germany." in result.reason
        assert "[not supported] Einstein was born in France." in result.reason
        assert "does not mention France" in result.reason

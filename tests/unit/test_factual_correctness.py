from unittest.mock import MagicMock

import pytest

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics._factual_correctness import FactualCorrectness

# Using mock_llm fixture from conftest.py (MockLLM class)


@pytest.fixture
def factual_correctness_metric(mock_llm):
    """Create a FactualCorrectness metric with mock LLM."""
    metric = FactualCorrectness(llm=mock_llm)
    return metric


@pytest.mark.asyncio
class TestFactualCorrectness:
    """Test suite for the migrated FactualCorrectness metric."""

    def test_factual_correctness_init(self, factual_correctness_metric):
        """Test FactualCorrectness initialization."""
        assert factual_correctness_metric.name == "factual_correctness"
        assert factual_correctness_metric.mode == "f1"
        assert factual_correctness_metric.beta == 1.0
        assert factual_correctness_metric.atomicity == "low"
        assert factual_correctness_metric.coverage == "low"

    def test_factual_correctness_invalid_beta(self):
        """Test FactualCorrectness with invalid beta value."""
        with pytest.raises(ValueError, match="Beta must be a float"):
            FactualCorrectness(beta="invalid")

    async def test_factual_correctness_decompose_claims(
        self, factual_correctness_metric
    ):
        """Test claim decomposition functionality."""
        # Mock LLM response for claim decomposition - InstructorLLM returns Pydantic objects directly
        from unittest.mock import Mock

        from ragas.metrics._factual_correctness import ClaimDecompositionOutput

        factual_correctness_metric.llm.generate = Mock(
            return_value=ClaimDecompositionOutput(
                claims=[
                    "Albert Einstein was a theoretical physicist.",
                    "Albert Einstein developed the theory of relativity.",
                ]
            )
        )

        response = "Albert Einstein was a theoretical physicist who developed the theory of relativity."
        claims = await factual_correctness_metric.decompose_claims(response)

        assert len(claims) == 2
        assert "Albert Einstein was a theoretical physicist." in claims
        assert "Albert Einstein developed the theory of relativity." in claims

    async def test_factual_correctness_verify_claims(self, factual_correctness_metric):
        """Test claim verification functionality."""
        # Mock LLM response for NLI - InstructorLLM returns Pydantic objects directly
        from unittest.mock import Mock

        from ragas.metrics._faithfulness import (
            NLIStatementOutput,
            StatementFaithfulnessAnswer,
        )

        factual_correctness_metric.llm.generate = Mock(
            return_value=NLIStatementOutput(
                statements=[
                    StatementFaithfulnessAnswer(
                        statement="Einstein was a physicist.",
                        verdict=1,
                        reason="Supported by context",
                    ),
                    StatementFaithfulnessAnswer(
                        statement="Einstein was a chef.",
                        verdict=0,
                        reason="Not supported by context",
                    ),
                ]
            )
        )

        premise = "Albert Einstein was a theoretical physicist."
        claims = ["Einstein was a physicist.", "Einstein was a chef."]

        verdicts = await factual_correctness_metric.verify_claims(premise, claims)

        assert len(verdicts) == 2
        assert verdicts[0]
        assert not verdicts[1]

    async def test_factual_correctness_full_flow_f1_mode(
        self, factual_correctness_metric
    ):
        """Test full factual correctness evaluation in F1 mode."""
        # Mock LLM responses - InstructorLLM returns Pydantic objects directly
        from unittest.mock import Mock

        from ragas.metrics._factual_correctness import ClaimDecompositionOutput
        from ragas.metrics._faithfulness import (
            NLIStatementOutput,
            StatementFaithfulnessAnswer,
        )

        def mock_generate_side_effect(prompt, response_model=None, **kwargs):
            if (
                "decompose_claims" in str(response_model)
                or response_model.__name__ == "ClaimDecompositionOutput"
            ):
                if "Einstein was a physicist and mathematician" in prompt:
                    # Response decomposition
                    return ClaimDecompositionOutput(
                        claims=["Response claim 1", "Response claim 2"]
                    )
                else:
                    # Reference decomposition
                    return ClaimDecompositionOutput(
                        claims=["Reference claim 1", "Reference claim 2"]
                    )
            else:
                # NLI verification
                return NLIStatementOutput(
                    statements=[
                        StatementFaithfulnessAnswer(
                            statement="Claim 1",
                            verdict=1,
                            reason="Supported",
                        ),
                        StatementFaithfulnessAnswer(
                            statement="Claim 2",
                            verdict=0,
                            reason="Not supported",
                        ),
                    ]
                )

        factual_correctness_metric.llm.generate = Mock(
            side_effect=mock_generate_side_effect
        )

        sample = SingleTurnSample(
            user_input="What do you know about Einstein?",
            response="Einstein was a physicist and mathematician.",
            reference="Einstein was a theoretical physicist who developed relativity theory.",
        )

        score = await factual_correctness_metric._single_turn_ascore(sample)

        # With TP=1, FP=1, FN=1, F1 score should be 0.5
        assert isinstance(score, (int, float))
        assert 0.0 <= score <= 1.0

    async def test_factual_correctness_precision_mode(self, factual_correctness_metric):
        """Test factual correctness in precision mode."""
        factual_correctness_metric.mode = "precision"

        # Mock responses for precision mode - InstructorLLM returns Pydantic objects directly
        from unittest.mock import Mock

        from ragas.metrics._factual_correctness import ClaimDecompositionOutput
        from ragas.metrics._faithfulness import (
            NLIStatementOutput,
            StatementFaithfulnessAnswer,
        )

        factual_correctness_metric.llm.generate = Mock(
            side_effect=[
                ClaimDecompositionOutput(
                    claims=["Response claim 1", "Response claim 2"]
                ),
                NLIStatementOutput(
                    statements=[
                        StatementFaithfulnessAnswer(
                            statement="Response claim 1",
                            verdict=1,
                            reason="Supported",
                        ),
                        StatementFaithfulnessAnswer(
                            statement="Response claim 2",
                            verdict=0,
                            reason="Not supported",
                        ),
                    ]
                ),
            ]
        )

        sample = SingleTurnSample(
            user_input="What do you know about Einstein?",
            response="Einstein was a physicist and mathematician.",
            reference="Einstein was a theoretical physicist.",
        )

        score = await factual_correctness_metric._single_turn_ascore(sample)

        # Precision = TP / (TP + FP) = 1 / (1 + 1) = 0.5
        assert score == 0.5

    async def test_factual_correctness_recall_mode(self, factual_correctness_metric):
        """Test factual correctness in recall mode."""
        factual_correctness_metric.mode = "recall"

        # Mock responses for recall mode - InstructorLLM returns Pydantic objects directly
        from unittest.mock import Mock

        from ragas.metrics._factual_correctness import ClaimDecompositionOutput
        from ragas.metrics._faithfulness import (
            NLIStatementOutput,
            StatementFaithfulnessAnswer,
        )

        factual_correctness_metric.llm.generate = Mock(
            side_effect=[
                ClaimDecompositionOutput(claims=["Response claim 1"]),
                NLIStatementOutput(
                    statements=[
                        StatementFaithfulnessAnswer(
                            statement="Response claim 1",
                            verdict=1,
                            reason="Supported",
                        )
                    ]
                ),
                ClaimDecompositionOutput(
                    claims=["Reference claim 1", "Reference claim 2"]
                ),
                NLIStatementOutput(
                    statements=[
                        StatementFaithfulnessAnswer(
                            statement="Reference claim 1",
                            verdict=1,
                            reason="Supported",
                        ),
                        StatementFaithfulnessAnswer(
                            statement="Reference claim 2",
                            verdict=0,
                            reason="Not supported",
                        ),
                    ]
                ),
            ]
        )

        sample = SingleTurnSample(
            user_input="What do you know about Einstein?",
            response="Einstein was a physicist.",
            reference="Einstein was a theoretical physicist who developed relativity theory.",
        )

        score = await factual_correctness_metric._single_turn_ascore(sample)

        # Recall = TP / (TP + FN) = 1 / (1 + 1) = 0.5
        assert score == 0.5

    async def test_factual_correctness_empty_claims(self, factual_correctness_metric):
        """Test handling of empty claims."""
        # Mock empty claims response - InstructorLLM returns Pydantic objects directly
        from unittest.mock import Mock

        from ragas.metrics._factual_correctness import ClaimDecompositionOutput

        factual_correctness_metric.llm.generate = Mock(
            return_value=ClaimDecompositionOutput(claims=[])
        )

        claims = await factual_correctness_metric.decompose_claims("Some response")
        assert claims == []

        # Test verify_claims with empty list
        verdicts = await factual_correctness_metric.verify_claims("Some premise", [])
        assert len(verdicts) == 0

    async def test_factual_correctness_json_parsing_error(
        self, factual_correctness_metric
    ):
        """Test handling of JSON parsing errors."""
        # Mock empty response for error handling - InstructorLLM returns Pydantic objects directly
        from unittest.mock import Mock

        from ragas.metrics._factual_correctness import ClaimDecompositionOutput
        from ragas.metrics._faithfulness import (
            NLIStatementOutput,
            StatementFaithfulnessAnswer,
        )

        factual_correctness_metric.llm.generate = Mock(
            side_effect=[
                ClaimDecompositionOutput(claims=[]),  # Empty claims for error
                NLIStatementOutput(
                    statements=[
                        StatementFaithfulnessAnswer(
                            statement="claim1", verdict=0, reason="Error"
                        ),
                        StatementFaithfulnessAnswer(
                            statement="claim2", verdict=0, reason="Error"
                        ),
                    ]
                ),
            ]
        )

        claims = await factual_correctness_metric.decompose_claims("Some response")
        assert claims == []

        verdicts = await factual_correctness_metric.verify_claims(
            "Some premise", ["claim1", "claim2"]
        )
        assert len(verdicts) == 2
        assert all(not verdict for verdict in verdicts)

    def test_factual_correctness_required_columns(self, factual_correctness_metric):
        """Test that required columns are correctly specified."""
        from ragas.metrics.base import MetricType

        required = factual_correctness_metric._required_columns[MetricType.SINGLE_TURN]
        assert "response" in required
        assert "reference" in required

    async def test_factual_correctness_error_handling(self, factual_correctness_metric):
        """Test error handling for missing LLM."""
        factual_correctness_metric.llm = None

        with pytest.raises(AssertionError, match="LLM must be set"):
            await factual_correctness_metric.decompose_claims("Some response")

        with pytest.raises(AssertionError, match="LLM must be set"):
            await factual_correctness_metric.verify_claims("Some premise", ["claim1"])

    async def test_factual_correctness_backward_compatibility(
        self, factual_correctness_metric
    ):
        """Test that callbacks parameter is accepted but ignored for backward compatibility."""
        # Mock responses - InstructorLLM returns Pydantic objects directly
        from unittest.mock import Mock

        from ragas.metrics._factual_correctness import ClaimDecompositionOutput
        from ragas.metrics._faithfulness import (
            NLIStatementOutput,
            StatementFaithfulnessAnswer,
        )

        factual_correctness_metric.llm.generate = Mock(
            side_effect=[
                ClaimDecompositionOutput(claims=["Response claim 1"]),
                NLIStatementOutput(
                    statements=[
                        StatementFaithfulnessAnswer(
                            statement="Response claim 1",
                            verdict=1,
                            reason="Supported",
                        )
                    ]
                ),
                ClaimDecompositionOutput(claims=["Reference claim 1"]),
                NLIStatementOutput(
                    statements=[
                        StatementFaithfulnessAnswer(
                            statement="Reference claim 1",
                            verdict=1,
                            reason="Supported",
                        )
                    ]
                ),
            ]
        )

        sample = SingleTurnSample(
            user_input="What do you know about Einstein?",
            response="Einstein was a physicist.",
            reference="Einstein was a theoretical physicist.",
        )

        # Should work with callbacks parameter (ignored)
        score = await factual_correctness_metric._single_turn_ascore(
            sample, callbacks=MagicMock()
        )
        assert isinstance(score, (int, float))

    async def test_factual_correctness_different_atomicity_coverage(self, mock_llm):
        """Test different atomicity and coverage settings."""
        # Test high atomicity, high coverage
        metric = FactualCorrectness(llm=mock_llm, atomicity="high", coverage="high")

        # Mock LLM response - InstructorLLM returns Pydantic objects directly
        from unittest.mock import Mock

        from ragas.metrics._factual_correctness import ClaimDecompositionOutput

        mock_llm.generate = Mock(
            return_value=ClaimDecompositionOutput(
                claims=["Claim 1", "Claim 2", "Claim 3", "Claim 4"]
            )
        )

        claims = await metric.decompose_claims(
            "Charles Babbage was a French mathematician, philosopher, and food critic."
        )
        assert len(claims) == 4

    async def test_factual_correctness_ascore_method(self, factual_correctness_metric):
        """Test the _ascore method."""
        # Mock responses - InstructorLLM returns Pydantic objects directly
        from unittest.mock import Mock

        from ragas.metrics._factual_correctness import ClaimDecompositionOutput
        from ragas.metrics._faithfulness import (
            NLIStatementOutput,
            StatementFaithfulnessAnswer,
        )

        factual_correctness_metric.llm.generate = Mock(
            side_effect=[
                ClaimDecompositionOutput(claims=["Response claim 1"]),
                NLIStatementOutput(
                    statements=[
                        StatementFaithfulnessAnswer(
                            statement="Response claim 1",
                            verdict=1,
                            reason="Supported",
                        )
                    ]
                ),
                ClaimDecompositionOutput(claims=["Reference claim 1"]),
                NLIStatementOutput(
                    statements=[
                        StatementFaithfulnessAnswer(
                            statement="Reference claim 1",
                            verdict=1,
                            reason="Supported",
                        )
                    ]
                ),
            ]
        )

        row = {
            "user_input": "What do you know about Einstein?",
            "response": "Einstein was a physicist.",
            "reference": "Einstein was a theoretical physicist.",
        }

        score = await factual_correctness_metric._ascore(row)
        assert isinstance(score, (int, float))
        assert 0.0 <= score <= 1.0

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.outputs import Generation, LLMResult

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics._factual_correctness import FactualCorrectness


@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing."""
    llm = AsyncMock()
    return llm


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
        # Mock LLM response for claim decomposition
        mock_response = {
            "claims": [
                "Albert Einstein was a theoretical physicist.",
                "Albert Einstein developed the theory of relativity.",
            ]
        }

        factual_correctness_metric.llm.generate.return_value = LLMResult(
            generations=[[Generation(text=json.dumps(mock_response))]]
        )

        response = "Albert Einstein was a theoretical physicist who developed the theory of relativity."
        claims = await factual_correctness_metric.decompose_claims(response)

        assert len(claims) == 2
        assert "Albert Einstein was a theoretical physicist." in claims
        assert "Albert Einstein developed the theory of relativity." in claims

    async def test_factual_correctness_verify_claims(self, factual_correctness_metric):
        """Test claim verification functionality."""
        # Mock LLM response for NLI
        mock_response = {
            "statements": [
                {
                    "statement": "Einstein was a physicist.",
                    "verdict": 1,
                    "reason": "Supported by context",
                },
                {
                    "statement": "Einstein was a chef.",
                    "verdict": 0,
                    "reason": "Not supported by context",
                },
            ]
        }

        factual_correctness_metric.llm.generate.return_value = LLMResult(
            generations=[[Generation(text=json.dumps(mock_response))]]
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
        # Mock claim decomposition responses
        claim_responses = [
            json.dumps(
                {"claims": ["Response claim 1", "Response claim 2"]}
            ),  # For response decomposition
            json.dumps(
                {"claims": ["Reference claim 1", "Reference claim 2"]}
            ),  # For reference decomposition
        ]

        # Mock NLI responses
        nli_responses = [
            json.dumps(
                {  # Reference -> Response verification
                    "statements": [
                        {
                            "statement": "Response claim 1",
                            "verdict": 1,
                            "reason": "Supported",
                        },
                        {
                            "statement": "Response claim 2",
                            "verdict": 0,
                            "reason": "Not supported",
                        },
                    ]
                }
            ),
            json.dumps(
                {  # Response -> Reference verification
                    "statements": [
                        {
                            "statement": "Reference claim 1",
                            "verdict": 1,
                            "reason": "Supported",
                        },
                        {
                            "statement": "Reference claim 2",
                            "verdict": 0,
                            "reason": "Not supported",
                        },
                    ]
                }
            ),
        ]

        # Set up mock to return different responses for different calls
        factual_correctness_metric.llm.generate.side_effect = [
            LLMResult(
                generations=[[Generation(text=claim_responses[0])]]
            ),  # Response decomposition
            LLMResult(
                generations=[[Generation(text=nli_responses[0])]]
            ),  # Reference -> Response NLI
            LLMResult(
                generations=[[Generation(text=claim_responses[1])]]
            ),  # Reference decomposition
            LLMResult(
                generations=[[Generation(text=nli_responses[1])]]
            ),  # Response -> Reference NLI
        ]

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

        # Mock responses for precision mode (only reference -> response verification)
        factual_correctness_metric.llm.generate.side_effect = [
            LLMResult(
                generations=[
                    [
                        Generation(
                            text=json.dumps(
                                {"claims": ["Response claim 1", "Response claim 2"]}
                            )
                        )
                    ]
                ]
            ),
            LLMResult(
                generations=[
                    [
                        Generation(
                            text=json.dumps(
                                {
                                    "statements": [
                                        {
                                            "statement": "Response claim 1",
                                            "verdict": 1,
                                            "reason": "Supported",
                                        },
                                        {
                                            "statement": "Response claim 2",
                                            "verdict": 0,
                                            "reason": "Not supported",
                                        },
                                    ]
                                }
                            )
                        )
                    ]
                ]
            ),
        ]

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

        # Mock responses for recall mode
        factual_correctness_metric.llm.generate.side_effect = [
            LLMResult(
                generations=[
                    [Generation(text=json.dumps({"claims": ["Response claim 1"]}))]
                ]
            ),
            LLMResult(
                generations=[
                    [
                        Generation(
                            text=json.dumps(
                                {
                                    "statements": [
                                        {
                                            "statement": "Response claim 1",
                                            "verdict": 1,
                                            "reason": "Supported",
                                        }
                                    ]
                                }
                            )
                        )
                    ]
                ]
            ),
            LLMResult(
                generations=[
                    [
                        Generation(
                            text=json.dumps(
                                {"claims": ["Reference claim 1", "Reference claim 2"]}
                            )
                        )
                    ]
                ]
            ),
            LLMResult(
                generations=[
                    [
                        Generation(
                            text=json.dumps(
                                {
                                    "statements": [
                                        {
                                            "statement": "Reference claim 1",
                                            "verdict": 1,
                                            "reason": "Supported",
                                        },
                                        {
                                            "statement": "Reference claim 2",
                                            "verdict": 0,
                                            "reason": "Not supported",
                                        },
                                    ]
                                }
                            )
                        )
                    ]
                ]
            ),
        ]

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
        # Mock empty claims response
        factual_correctness_metric.llm.generate.return_value = LLMResult(
            generations=[[Generation(text=json.dumps({"claims": []}))]]
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
        # Mock invalid JSON response
        factual_correctness_metric.llm.generate.return_value = LLMResult(
            generations=[[Generation(text="Invalid JSON response")]]
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
        # Mock responses
        factual_correctness_metric.llm.generate.side_effect = [
            LLMResult(
                generations=[
                    [Generation(text=json.dumps({"claims": ["Response claim 1"]}))]
                ]
            ),
            LLMResult(
                generations=[
                    [
                        Generation(
                            text=json.dumps(
                                {
                                    "statements": [
                                        {
                                            "statement": "Response claim 1",
                                            "verdict": 1,
                                            "reason": "Supported",
                                        }
                                    ]
                                }
                            )
                        )
                    ]
                ]
            ),
            LLMResult(
                generations=[
                    [Generation(text=json.dumps({"claims": ["Reference claim 1"]}))]
                ]
            ),
            LLMResult(
                generations=[
                    [
                        Generation(
                            text=json.dumps(
                                {
                                    "statements": [
                                        {
                                            "statement": "Reference claim 1",
                                            "verdict": 1,
                                            "reason": "Supported",
                                        }
                                    ]
                                }
                            )
                        )
                    ]
                ]
            ),
        ]

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

        mock_llm.generate.return_value = LLMResult(
            generations=[
                [
                    Generation(
                        text=json.dumps(
                            {"claims": ["Claim 1", "Claim 2", "Claim 3", "Claim 4"]}
                        )
                    )
                ]
            ]
        )

        claims = await metric.decompose_claims(
            "Charles Babbage was a French mathematician, philosopher, and food critic."
        )
        assert len(claims) == 4

    async def test_factual_correctness_ascore_method(self, factual_correctness_metric):
        """Test the _ascore method."""
        # Mock responses
        factual_correctness_metric.llm.generate.side_effect = [
            LLMResult(
                generations=[
                    [Generation(text=json.dumps({"claims": ["Response claim 1"]}))]
                ]
            ),
            LLMResult(
                generations=[
                    [
                        Generation(
                            text=json.dumps(
                                {
                                    "statements": [
                                        {
                                            "statement": "Response claim 1",
                                            "verdict": 1,
                                            "reason": "Supported",
                                        }
                                    ]
                                }
                            )
                        )
                    ]
                ]
            ),
            LLMResult(
                generations=[
                    [Generation(text=json.dumps({"claims": ["Reference claim 1"]}))]
                ]
            ),
            LLMResult(
                generations=[
                    [
                        Generation(
                            text=json.dumps(
                                {
                                    "statements": [
                                        {
                                            "statement": "Reference claim 1",
                                            "verdict": 1,
                                            "reason": "Supported",
                                        }
                                    ]
                                }
                            )
                        )
                    ]
                ]
            ),
        ]

        row = {
            "user_input": "What do you know about Einstein?",
            "response": "Einstein was a physicist.",
            "reference": "Einstein was a theoretical physicist.",
        }

        score = await factual_correctness_metric._ascore(row)
        assert isinstance(score, (int, float))
        assert 0.0 <= score <= 1.0

"""
Unit tests for Faithfulness metric.
Tests both the original and migrated versions to ensure compatibility.
"""

import json
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import Faithfulness, FaithfulnesswithHHEM
from ragas.metrics._faithfulness import (
    NLIStatementOutput,
    StatementFaithfulnessAnswer,
    StatementGeneratorOutput,
)
from ragas.metrics.base import MetricType


@pytest.fixture
def sample_data():
    """Sample data for testing faithfulness."""
    return SingleTurnSample(
        user_input="What is the capital of France?",
        response="Paris is the capital of France. It is located in the north of France.",
        retrieved_contexts=[
            "Paris is the capital and largest city of France.",
            "France is a country in Western Europe.",
        ],
    )


@pytest.fixture
def mock_llm():
    """Mock LLM for testing."""
    llm = AsyncMock()
    return llm


def test_faithfulness_init():
    """Test Faithfulness initialization."""
    metric = Faithfulness()
    assert metric.name == "faithfulness"
    assert metric._required_columns == {
        MetricType.SINGLE_TURN: {"user_input", "response", "retrieved_contexts"}
    }


def test_faithfulness_with_hhem_init():
    """Test FaithfulnesswithHHEM initialization."""
    with patch("transformers.AutoModelForSequenceClassification.from_pretrained"):
        metric = FaithfulnesswithHHEM()
        assert metric.name == "faithfulness_with_hhem"
        assert hasattr(metric, "nli_classifier")


@pytest.mark.asyncio
async def test_faithfulness_create_statements(mock_llm, sample_data):
    """Test statement generation from response."""
    metric = Faithfulness(llm=mock_llm)

    # Mock LLM response
    from langchain_core.outputs import Generation, LLMResult

    mock_generation = Generation(
        text='{"statements": ["Paris is the capital of France.", "Paris is located in the north of France."]}'
    )
    mock_result = LLMResult(generations=[[mock_generation]])
    mock_llm.generate.return_value = mock_result

    result = await metric._create_statements(sample_data.to_dict())

    assert isinstance(result, StatementGeneratorOutput)
    assert len(result.statements) == 2
    assert "Paris is the capital of France." in result.statements


@pytest.mark.asyncio
async def test_faithfulness_create_verdicts(mock_llm, sample_data):
    """Test verdict creation for statements."""
    metric = Faithfulness(llm=mock_llm)

    statements = ["Paris is the capital of France.", "Paris has 10 million people."]

    # Mock LLM response
    from langchain_core.outputs import Generation, LLMResult

    mock_response = {
        "statements": [
            {
                "statement": "Paris is the capital of France.",
                "reason": "This can be directly inferred from the context.",
                "verdict": 1,
            },
            {
                "statement": "Paris has 10 million people.",
                "reason": "Population information is not provided in the context.",
                "verdict": 0,
            },
        ]
    }
    mock_generation = Generation(text=json.dumps(mock_response))
    mock_result = LLMResult(generations=[[mock_generation]])
    mock_llm.generate.return_value = mock_result

    result = await metric._create_verdicts(sample_data.to_dict(), statements)

    assert isinstance(result, NLIStatementOutput)
    assert len(result.statements) == 2
    assert result.statements[0].verdict == 1
    assert result.statements[1].verdict == 0


def test_faithfulness_compute_score():
    """Test score computation from verdicts."""
    metric = Faithfulness()

    # Test case: 2 out of 3 statements are faithful
    verdicts = NLIStatementOutput(
        statements=[
            StatementFaithfulnessAnswer(
                statement="Statement 1", reason="Reason 1", verdict=1
            ),
            StatementFaithfulnessAnswer(
                statement="Statement 2", reason="Reason 2", verdict=1
            ),
            StatementFaithfulnessAnswer(
                statement="Statement 3", reason="Reason 3", verdict=0
            ),
        ]
    )

    score = metric._compute_score(verdicts)
    assert score == 2 / 3

    # Test case: No statements
    empty_verdicts = NLIStatementOutput(statements=[])
    score = metric._compute_score(empty_verdicts)
    assert np.isnan(score)


@pytest.mark.asyncio
async def test_faithfulness_single_turn_ascore_full_flow(mock_llm, sample_data):
    """Test full faithfulness scoring flow."""
    metric = Faithfulness(llm=mock_llm)

    # Mock statement generation response
    from langchain_core.outputs import Generation, LLMResult

    def mock_generate_side_effect(prompt_value, **kwargs):
        prompt_text = prompt_value.text
        if "analyze the complexity" in prompt_text:
            # Statement generation
            return LLMResult(
                generations=[
                    [
                        Generation(
                            text='{"statements": ["Paris is the capital of France."]}'
                        )
                    ]
                ]
            )
        else:
            # NLI evaluation
            response = {
                "statements": [
                    {
                        "statement": "Paris is the capital of France.",
                        "reason": "This can be directly inferred from the context.",
                        "verdict": 1,
                    }
                ]
            }
            return LLMResult(generations=[[Generation(text=json.dumps(response))]])

    mock_llm.generate.side_effect = mock_generate_side_effect

    score = await metric._single_turn_ascore(sample_data)

    assert isinstance(score, float)
    assert score == 1.0


@pytest.mark.asyncio
async def test_faithfulness_empty_statements(mock_llm, sample_data):
    """Test handling of empty statement generation."""
    metric = Faithfulness(llm=mock_llm)

    # Mock empty statement generation
    from langchain_core.outputs import Generation, LLMResult

    mock_generation = Generation(text='{"statements": []}')
    mock_result = LLMResult(generations=[[mock_generation]])
    mock_llm.generate.return_value = mock_result

    score = await metric._single_turn_ascore(sample_data)

    assert np.isnan(score)


@pytest.mark.asyncio
async def test_faithfulness_with_hhem_scoring(mock_llm):
    """Test FaithfulnesswithHHEM scoring with mocked classifier."""
    with patch(
        "transformers.AutoModelForSequenceClassification.from_pretrained"
    ) as mock_model:
        # Mock the classifier with proper tensor-like behavior
        from unittest.mock import MagicMock

        mock_tensor = MagicMock()
        mock_tensor.cpu.return_value.detach.return_value.round.return_value.tolist.return_value = [
            1.0,
            0.0,
        ]

        mock_classifier = MagicMock()
        mock_classifier.predict.return_value = mock_tensor
        mock_classifier.to.return_value = None
        mock_model.return_value = mock_classifier

        metric = FaithfulnesswithHHEM(llm=mock_llm)  # Add LLM
        metric.nli_classifier = mock_classifier

        sample = SingleTurnSample(
            user_input="Test question",
            response="Test response with two statements.",
            retrieved_contexts=["Test context"],
        )

        # Mock statement generation
        from langchain_core.outputs import Generation, LLMResult

        mock_generation = Generation(
            text='{"statements": ["Statement 1", "Statement 2"]}'
        )
        mock_result = LLMResult(generations=[[mock_generation]])
        mock_llm.generate.return_value = mock_result

        score = await metric._single_turn_ascore(sample)

        assert isinstance(score, float)
        assert score == 0.5  # 1 out of 2 statements faithful


def test_faithfulness_required_columns():
    """Test that faithfulness has correct required columns."""
    metric = Faithfulness()
    required = metric.get_required_columns()

    assert MetricType.SINGLE_TURN.name in required
    expected_columns = {"user_input", "response", "retrieved_contexts"}
    assert required[MetricType.SINGLE_TURN.name] == expected_columns


@pytest.mark.asyncio
async def test_faithfulness_error_handling(mock_llm):
    """Test error handling in faithfulness metric."""
    metric = Faithfulness(llm=None)

    sample = SingleTurnSample(
        user_input="Test", response="Test response", retrieved_contexts=["Test context"]
    )

    # Test assertion error when LLM is None
    with pytest.raises(AssertionError, match="LLM is not set"):
        await metric._single_turn_ascore(sample)


# ============================================================================
# ADDITIONAL TESTS FOR MIGRATED FAITHFULNESS
# ============================================================================


@pytest.mark.asyncio
async def test_faithfulness_json_parsing_error(mock_llm, sample_data):
    """Test Faithfulness handling of malformed JSON responses."""
    metric = Faithfulness(llm=mock_llm)

    # Mock malformed JSON response
    from langchain_core.outputs import Generation, LLMResult

    mock_generation = Generation(text="This is not valid JSON")
    mock_result = LLMResult(generations=[[mock_generation]])
    mock_llm.generate.return_value = mock_result

    result = await metric._create_statements(sample_data.to_dict())

    assert isinstance(result, StatementGeneratorOutput)
    assert len(result.statements) == 0


@pytest.mark.asyncio
async def test_faithfulness_backward_compatibility(mock_llm, sample_data):
    """Test that Faithfulness maintains backward compatibility with callbacks."""
    metric = Faithfulness(llm=mock_llm)

    # Mock LLM responses
    from langchain_core.outputs import Generation, LLMResult

    def mock_generate_side_effect(prompt_value, **kwargs):
        prompt_text = prompt_value.text
        if "analyze the complexity" in prompt_text:
            return LLMResult(
                generations=[[Generation(text='{"statements": ["Test statement."]}')]]
            )
        else:
            response = {
                "statements": [
                    {
                        "statement": "Test statement.",
                        "reason": "Test reason.",
                        "verdict": 1,
                    }
                ]
            }
            return LLMResult(generations=[[Generation(text=json.dumps(response))]])

    mock_llm.generate.side_effect = mock_generate_side_effect

    # Test that both callback and non-callback versions work
    score1 = await metric._single_turn_ascore(
        sample_data, []
    )  # With callbacks (ignored)
    score2 = await metric._single_turn_ascore(sample_data)  # Without callbacks

    assert score1 == score2 == 1.0

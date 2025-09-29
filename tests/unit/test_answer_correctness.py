"""
Unit tests for AnswerCorrectness metric.
Tests the migrated version without LangChain dependencies.
"""

import pytest

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import AnswerCorrectness
from ragas.metrics._answer_correctness import (
    ClassificationWithReason,
    StatementsWithReason,
)
from ragas.metrics.base import MetricType


@pytest.fixture
def sample_data():
    """Sample data for testing answer correctness."""
    return SingleTurnSample(
        user_input="What is the capital of France?",
        response="Paris is the capital of France. It is a beautiful city.",
        reference="Paris is the capital of France. It is located in northern France.",
    )


# Using mock_llm fixture from conftest.py (MockLLM class)


@pytest.fixture
def mock_embeddings():
    """Mock embeddings for testing."""
    from unittest.mock import AsyncMock

    embeddings = AsyncMock()
    embeddings.aembed_text = AsyncMock()
    return embeddings


def test_answer_correctness_init():
    """Test AnswerCorrectness initialization."""
    metric = AnswerCorrectness()
    assert metric.name == "answer_correctness"
    assert metric._required_columns == {
        MetricType.SINGLE_TURN: {"user_input", "response", "reference"}
    }
    assert metric.weights == [0.75, 0.25]
    assert metric.beta == 1.0


def test_answer_correctness_invalid_weights():
    """Test AnswerCorrectness with invalid weights."""
    # Test wrong number of weights
    with pytest.raises(ValueError, match="Expects a list of two weights"):
        AnswerCorrectness(weights=[0.5])

    # Test all zero weights
    with pytest.raises(ValueError, match="At least one weight must be non-zero"):
        AnswerCorrectness(weights=[0.0, 0.0])

    # Test negative weights
    with pytest.raises(ValueError, match="Weights must be non-negative"):
        AnswerCorrectness(weights=[-0.5, 0.5])


def test_answer_correctness_invalid_beta():
    """Test AnswerCorrectness with invalid beta."""
    with pytest.raises(ValueError, match="Beta must be a float"):
        AnswerCorrectness(beta="invalid")


@pytest.mark.asyncio
async def test_answer_correctness_create_statements(mock_llm, sample_data):
    """Test statement generation in AnswerCorrectness."""
    metric = AnswerCorrectness(llm=mock_llm)

    # Mock LLM response - InstructorLLM returns Pydantic objects directly
    from unittest.mock import Mock

    from ragas.metrics._faithfulness import StatementGeneratorOutput

    # Replace the generate method with a mock that returns our test data
    mock_llm.generate = Mock(
        return_value=StatementGeneratorOutput(
            statements=["Paris is the capital of France.", "Paris is a beautiful city."]
        )
    )

    result = await metric._create_simplified_statements(
        "What is the capital of France?",
        "Paris is the capital of France. It is a beautiful city.",
    )

    assert isinstance(result, StatementGeneratorOutput)
    assert len(result.statements) == 2
    assert "Paris is the capital of France." in result.statements


@pytest.mark.asyncio
async def test_answer_correctness_classify_statements(mock_llm, sample_data):
    """Test statement classification in AnswerCorrectness."""
    metric = AnswerCorrectness(llm=mock_llm)

    answer = ["Paris is the capital of France.", "Paris has 10 million people."]
    ground_truth = [
        "Paris is the capital of France.",
        "Paris is located in northern France.",
    ]

    # Mock LLM response - InstructorLLM returns Pydantic objects directly
    from unittest.mock import Mock

    from ragas.metrics._answer_correctness import (
        ClassificationWithReason,
        StatementsWithReason,
    )

    # Replace the generate method with a mock that returns our test data
    mock_llm.generate = Mock(
        return_value=ClassificationWithReason(
            TP=[
                StatementsWithReason(
                    statement="Paris is the capital of France.",
                    reason="This statement is directly supported by the ground truth.",
                )
            ],
            FP=[
                StatementsWithReason(
                    statement="Paris has 10 million people.",
                    reason="Population information is not provided in the ground truth.",
                )
            ],
            FN=[
                StatementsWithReason(
                    statement="Paris is located in northern France.",
                    reason="This location information is not mentioned in the answer.",
                )
            ],
        )
    )

    result = await metric._classify_statements(
        "What is the capital of France?", answer, ground_truth
    )

    assert isinstance(result, ClassificationWithReason)
    assert len(result.TP) == 1
    assert len(result.FP) == 1
    assert len(result.FN) == 1
    assert result.TP[0].statement == "Paris is the capital of France."


def test_answer_correctness_compute_statement_presence():
    """Test statement presence computation."""
    metric = AnswerCorrectness()

    # Test case: 1 TP, 1 FP, 1 FN
    classification = ClassificationWithReason(
        TP=[StatementsWithReason(statement="TP statement", reason="TP reason")],
        FP=[StatementsWithReason(statement="FP statement", reason="FP reason")],
        FN=[StatementsWithReason(statement="FN statement", reason="FN reason")],
    )

    score = metric._compute_statement_presence(classification)
    # F1 score with TP=1, FP=1, FN=1 should be 0.5
    assert isinstance(score, float)
    assert 0.4 < score < 0.6  # Approximate check for F1 score


@pytest.mark.asyncio
async def test_answer_correctness_full_flow_factuality_only(mock_llm, sample_data):
    """Test full AnswerCorrectness scoring flow with factuality only."""
    metric = AnswerCorrectness(llm=mock_llm, weights=[1.0, 0.0])  # Only factuality

    # Mock statement generation response - InstructorLLM returns Pydantic objects directly
    from unittest.mock import Mock

    from ragas.metrics._answer_correctness import (
        ClassificationWithReason,
        StatementsWithReason,
    )
    from ragas.metrics._faithfulness import StatementGeneratorOutput

    def mock_generate_side_effect(prompt, response_model=None, **kwargs):
        if "analyze the complexity" in prompt:
            # Statement generation
            return StatementGeneratorOutput(
                statements=["Paris is the capital of France."]
            )
        else:
            # Classification
            return ClassificationWithReason(
                TP=[
                    StatementsWithReason(
                        statement="Paris is the capital of France.",
                        reason="This statement is directly supported by the ground truth.",
                    )
                ],
                FP=[],
                FN=[],
            )

    mock_llm.generate = Mock(side_effect=mock_generate_side_effect)

    score = await metric._single_turn_ascore(sample_data)

    assert isinstance(score, float)
    assert score == 1.0  # Perfect factuality score


@pytest.mark.asyncio
async def test_answer_correctness_full_flow_with_similarity(
    mock_llm, mock_embeddings, sample_data
):
    """Test full AnswerCorrectness scoring flow with both factuality and similarity."""
    metric = AnswerCorrectness(
        llm=mock_llm, embeddings=mock_embeddings, weights=[0.5, 0.5]
    )

    # Initialize the metric to set up answer_similarity
    from ragas.run_config import RunConfig

    metric.init(RunConfig())

    # Mock statement generation and classification - InstructorLLM returns Pydantic objects directly
    from unittest.mock import Mock

    from ragas.metrics._answer_correctness import (
        ClassificationWithReason,
        StatementsWithReason,
    )
    from ragas.metrics._faithfulness import StatementGeneratorOutput

    def mock_generate_side_effect(prompt, response_model=None, **kwargs):
        if "analyze the complexity" in prompt:
            return StatementGeneratorOutput(
                statements=["Paris is the capital of France."]
            )
        else:
            return ClassificationWithReason(
                TP=[
                    StatementsWithReason(
                        statement="Paris is the capital of France.",
                        reason="Supported by ground truth.",
                    )
                ],
                FP=[],
                FN=[],
            )

    mock_llm.generate = Mock(side_effect=mock_generate_side_effect)

    # Mock embeddings for similarity calculation
    mock_embeddings.aembed_text.side_effect = [
        [0.1, 0.2, 0.3],  # reference embedding
        [0.1, 0.2, 0.3],  # response embedding (same for perfect similarity)
    ]

    score = await metric._single_turn_ascore(sample_data)

    assert isinstance(score, float)
    assert (
        0.9 < score <= 1.0
    )  # High score due to perfect factuality and high similarity


@pytest.mark.asyncio
async def test_answer_correctness_empty_statements(mock_llm, sample_data):
    """Test AnswerCorrectness handling of empty statement generation."""
    metric = AnswerCorrectness(llm=mock_llm, weights=[1.0, 0.0])

    # Mock empty statement generation for both response and reference
    from unittest.mock import Mock

    from ragas.metrics._faithfulness import StatementGeneratorOutput

    mock_llm.generate = Mock(return_value=StatementGeneratorOutput(statements=[]))

    score = await metric._single_turn_ascore(sample_data)

    assert isinstance(score, float)
    assert score == 1.0  # Should return 1.0 when no statements are generated


@pytest.mark.asyncio
async def test_answer_correctness_json_parsing_error(mock_llm, sample_data):
    """Test AnswerCorrectness handling of malformed JSON responses."""
    metric = AnswerCorrectness(llm=mock_llm, weights=[1.0, 0.0])

    # Mock response - InstructorLLM ensures valid Pydantic objects (no JSON parsing errors possible)
    from unittest.mock import Mock

    from ragas.metrics._faithfulness import StatementGeneratorOutput

    mock_llm.generate = Mock(return_value=StatementGeneratorOutput(statements=[]))

    result = await metric._create_simplified_statements("Test question", "Test answer")

    assert isinstance(result, StatementGeneratorOutput)
    assert len(result.statements) == 0


@pytest.mark.asyncio
async def test_answer_correctness_backward_compatibility(mock_llm, sample_data):
    """Test that AnswerCorrectness maintains backward compatibility with callbacks."""
    metric = AnswerCorrectness(llm=mock_llm, weights=[1.0, 0.0])

    # Mock LLM responses - InstructorLLM returns Pydantic objects directly
    from unittest.mock import Mock

    from ragas.metrics._answer_correctness import (
        ClassificationWithReason,
        StatementsWithReason,
    )
    from ragas.metrics._faithfulness import StatementGeneratorOutput

    def mock_generate_side_effect(prompt, response_model=None, **kwargs):
        if "analyze the complexity" in prompt:
            return StatementGeneratorOutput(statements=["Test statement."])
        else:
            return ClassificationWithReason(
                TP=[
                    StatementsWithReason(
                        statement="Test statement.", reason="Test reason."
                    )
                ],
                FP=[],
                FN=[],
            )

    mock_llm.generate = Mock(side_effect=mock_generate_side_effect)

    # Test that both callback and non-callback versions work
    score1 = await metric._single_turn_ascore(
        sample_data, []
    )  # With callbacks (ignored)
    score2 = await metric._single_turn_ascore(sample_data)  # Without callbacks

    assert score1 == score2 == 1.0


def test_answer_correctness_required_columns():
    """Test that answer correctness has correct required columns."""
    metric = AnswerCorrectness()
    required = metric.get_required_columns()

    assert MetricType.SINGLE_TURN.name in required
    expected_columns = {"user_input", "response", "reference"}
    assert required[MetricType.SINGLE_TURN.name] == expected_columns


@pytest.mark.asyncio
async def test_answer_correctness_error_handling():
    """Test error handling in answer correctness metric."""
    metric = AnswerCorrectness(llm=None)

    sample = SingleTurnSample(
        user_input="Test", response="Test response", reference="Test reference"
    )

    # Test assertion error when LLM is None
    with pytest.raises(AssertionError, match="LLM must be set"):
        await metric._single_turn_ascore(sample)

import typing as t
from dataclasses import dataclass
import pytest
from pydantic import BaseModel

from ragas_experimental.metric.base import Metric
from ragas_experimental.metric import MetricResult


class MetricResponseModel(BaseModel):
    result: int
    reason: t.Optional[str] = None


@dataclass
class CustomMetric(Metric):
    """Custom metric implementation for testing."""

    def __post_init__(self):
        super().__post_init__()
        self._response_model = MetricResponseModel
        
    def get_correlation(self, gold_labels: t.List[str], predictions: t.List[str]) -> float:
        
        return 0.0  # Placeholder for correlation logic


@pytest.fixture
def mock_llm(mock_llm):
    """Use the mock LLM from conftest."""
    return mock_llm


def test_metric_creation():
    """Test creating a custom metric."""
    metric = CustomMetric(name="test_metric", prompt="What is the result of {input}?")

    assert metric.name == "test_metric"
    assert isinstance(metric.prompt, str) or hasattr(metric.prompt, "format")


def test_metric_get_variables():
    """Test extracting variables from prompt template."""
    metric = CustomMetric(
        name="test_metric",
        prompt="Evaluate the {question} given the {context} and {answer}",
    )

    variables = metric.get_variables()
    expected_vars = ["question", "context", "answer"]

    assert set(variables) == set(expected_vars)


def test_metric_score_single(mock_llm):
    """Test scoring with a single input."""
    metric = CustomMetric(name="test_metric", prompt="What is the result of {input}?")

    # Mock the LLM to return a valid response
    def mock_generate(prompt, response_model):
        return response_model(result=1, reason="test reason")

    mock_llm.generate = mock_generate

    result = metric.score(llm=mock_llm, input="test")

    assert isinstance(result, MetricResult)
    assert result.traces is not None
    assert "input" in result.traces


@pytest.mark.asyncio
async def test_metric_async_score(mock_llm):
    """Test async scoring functionality."""
    metric = CustomMetric(name="test_metric", prompt="What is the result of {input}?")

    # Mock the async LLM method
    async def mock_agenerate(prompt, response_model):
        return response_model(result=1, reason="test reason")

    mock_llm.agenerate = mock_agenerate

    result = await metric.ascore(llm=mock_llm, input="test")

    assert isinstance(result, MetricResult)
    assert result.traces is not None


def test_metric_response_model():
    """Test that metric has correct response model."""
    metric = CustomMetric(name="test_metric", prompt="What is the result of {input}?")

    assert metric._response_model == MetricResponseModel


def test_metric_prompt_conversion():
    """Test that string prompts are converted to Prompt objects."""
    metric = CustomMetric(name="test_metric", prompt="What is the result of {input}?")

    # After __post_init__, prompt should be converted to Prompt object
    assert hasattr(metric.prompt, "format")


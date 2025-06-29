import tempfile
import typing as t
import pytest
from unittest.mock import Mock
from dataclasses import dataclass, field
from ragas_experimental.project.core import Project
from ragas_experimental.model.pydantic_model import ExtendedPydanticBaseModel as BaseModel
from ragas_experimental.metric import MetricResult
from ragas_experimental.metric.base import Metric


class EvaluationData(BaseModel):
    """Model for evaluation data."""
    question: str
    context: str
    answer: str
    ground_truth: str


class EvaluationResult(BaseModel):
    """Model for evaluation results."""
    result: float
    reason: str


@dataclass
class IntegrationMetric(Metric):
    """Simple metric for integration testing."""
    
    def __post_init__(self):
        super().__post_init__()
        self._response_model = EvaluationResult
        
    def get_correlation(self, gold_label, predictions) -> float:
        return super().get_correlation(gold_label, predictions)


@pytest.fixture
def temp_project():
    """Create a temporary project for integration testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        project = Project.create(
            name="integration_test_project",
            description="Project for integration testing",
            backend="local/csv",
            root_dir=temp_dir
        )
        yield project


@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing."""
    llm = Mock()
    
    def mock_generate(prompt, response_model):
        return response_model(result=0.8, reason="Mock evaluation")
    
    llm.generate = mock_generate
    return llm


def test_full_evaluation_workflow(temp_project, mock_llm):
    """Test a complete evaluation workflow with project, dataset, and metrics."""
    
    # 1. Create a dataset
    dataset = temp_project.create_dataset(
        name="evaluation_dataset",
        model=EvaluationData
    )
    
    # 2. Add evaluation data
    eval_data = [
        EvaluationData(
            question="What is the capital of France?",
            context="France is a country in Europe. Its capital is Paris.",
            answer="Paris",
            ground_truth="Paris"
        ),
        EvaluationData(
            question="What is 2+2?",
            context="Basic arithmetic operations.",
            answer="4",
            ground_truth="4"
        )
    ]
    
    for data in eval_data:
        dataset.append(data)
    
    # 3. Create a metric
    metric = IntegrationMetric(
        name="integration_metric",
        prompt="Evaluate if the answer '{answer}' is correct given the question '{question}' and context '{context}'. Ground truth: '{ground_truth}'"
    )
    
    # 4. Run evaluation on dataset
    results = []
    for entry in dataset:
        result = metric.score(
            llm=mock_llm,
            question=entry.question,
            context=entry.context,
            answer=entry.answer,
            ground_truth=entry.ground_truth
        )
        results.append(result)
    
    # 5. Verify results
    assert len(results) == 2
    assert all(isinstance(result, MetricResult) for result in results)
    assert all(result.result == 0.8 for result in results)  # Mock always returns 0.8


def test_project_dataset_persistence(temp_project):
    """Test that data persists across dataset operations."""
    
    # Create dataset and add data
    dataset = temp_project.create_dataset(
        name="persistence_test",
        model=EvaluationData
    )
    
    test_data = EvaluationData(
        question="Test question",
        context="Test context", 
        answer="Test answer",
        ground_truth="Test ground truth"
    )
    
    dataset.append(test_data)
    assert len(dataset) == 1
    
    # Load dataset again (simulates persistence)
    dataset.load()
    assert len(dataset) == 1
    
    # Verify data integrity
    loaded_data = dataset[0]
    assert loaded_data.question == "Test question"
    assert loaded_data.context == "Test context"
    assert loaded_data.answer == "Test answer"
    assert loaded_data.ground_truth == "Test ground truth"


def test_batch_evaluation_workflow(temp_project, mock_llm):
    """Test batch evaluation across multiple entries."""
    
    # Create dataset with multiple entries
    dataset = temp_project.create_dataset(
        name="batch_evaluation",
        model=EvaluationData
    )
    
    # Add multiple evaluation entries
    for i in range(5):
        dataset.append(EvaluationData(
            question=f"Question {i}",
            context=f"Context {i}",
            answer=f"Answer {i}",
            ground_truth=f"Ground truth {i}"
        ))
    
    # Create metric
    metric = IntegrationMetric(
        name="batch_metric",
        prompt="Evaluate: {question} with context: {context} -> {answer} vs ground_truth: {ground_truth}"
    )
    
    # Run individual evaluations (since batch_score doesn't exist in the real API)
    batch_results = []
    for entry in dataset:
        result = metric.score(
            llm=mock_llm,
            question=entry.question,
            context=entry.context,
            answer=entry.answer,
            ground_truth=entry.ground_truth
        )
        batch_results.append(result)
    
    # Verify batch results
    assert len(batch_results) == 5
    assert all(isinstance(result, MetricResult) for result in batch_results)


def test_dataset_modification_workflow(temp_project):
    """Test modifying dataset entries and persistence."""
    
    dataset = temp_project.create_dataset(
        name="modification_test",
        model=EvaluationData
    )
    
    # Add initial data
    initial_data = EvaluationData(
        question="Initial question",
        context="Initial context",
        answer="Initial answer", 
        ground_truth="Initial ground truth"
    )
    dataset.append(initial_data)
    
    # Modify the entry
    entry = dataset[0]
    entry.answer = "Modified answer"
    dataset.save(entry)
    
    # Verify modification persisted
    assert dataset[0].answer == "Modified answer"
    
    # Load and verify persistence
    dataset.load()
    assert dataset[0].answer == "Modified answer"
    assert dataset[0].question == "Initial question"  # Other fields unchanged


def test_metric_variable_extraction_integration(mock_llm):
    """Test that metrics can extract variables from complex prompts."""
    
    metric = IntegrationMetric(
        name="variable_test",
        prompt="Given the question: '{question}', context: '{context}', and answer: '{answer}', evaluate against ground truth: '{ground_truth}'. Consider the difficulty: '{difficulty}' and domain: '{domain}'."
    )
    
    variables = metric.get_variables()
    expected_vars = {"question", "context", "answer", "ground_truth", "difficulty", "domain"}
    
    assert set(variables) == expected_vars


@pytest.mark.asyncio
async def test_async_evaluation_integration(temp_project):
    """Test async evaluation workflow."""
    
    # Mock async LLM
    async_llm = Mock()
    
    async def mock_agenerate(prompt, response_model):
        return response_model(result=0.9, reason="Async mock evaluation")
    
    async_llm.agenerate = mock_agenerate
    
    # Create metric
    metric = IntegrationMetric(
        name="async_metric",
        prompt="Async evaluate: {question} -> {answer}"
    )
    
    # Test async scoring
    result = await metric.ascore(
        llm=async_llm,
        question="Test question",
        answer="Test answer"
    )
    
    assert isinstance(result, MetricResult)
    assert result.result == 0.9
import pytest
import asyncio
from ragas.run_config import RunConfig
from ragas.metrics.base import Metric, SingleTurnMetric
from ragas.dataset_schema import SingleTurnSample
from typing import Optional, List


class TestTimeoutConfig:
    """Test the timeout configuration in RunConfig."""

    def test_default_timeout(self):
        """Test that the default timeout is set to 300 seconds."""
        config = RunConfig()
        assert config.timeout == 300, "Default timeout should be 300 seconds"

    def test_custom_timeout(self):
        """Test that a custom timeout can be set."""
        config = RunConfig(timeout=500)
        assert config.timeout == 500, "Custom timeout should be respected"


class SlowMetric(SingleTurnMetric):
    """A test metric that simulates slow processing."""
    
    name = "slow_metric"
    
    def __init__(self, sleep_time: float = 0.1):
        super().__init__()
        self.sleep_time = sleep_time
    
    def init(self):
        """Initialize the metric."""
        pass
    
    async def _single_turn_ascore(self, sample: SingleTurnSample, callbacks=None) -> float:
        """Simulate slow processing by sleeping."""
        await asyncio.sleep(self.sleep_time)
        return 1.0


@pytest.mark.asyncio
async def test_metric_timeout():
    """Test that the timeout is applied to metric scoring."""
    # Create a sample
    sample = SingleTurnSample(
        question="Test question",
        answer="Test answer",
        contexts=["Test context"]
    )
    
    # Create a slow metric
    slow_metric = SlowMetric(sleep_time=0.2)
    
    # Test with sufficient timeout
    score = await slow_metric.single_turn_ascore(sample, timeout=0.5)
    assert score == 1.0, "Metric should complete with sufficient timeout"
    
    # Test with insufficient timeout
    with pytest.raises(asyncio.TimeoutError):
        await slow_metric.single_turn_ascore(sample, timeout=0.1)
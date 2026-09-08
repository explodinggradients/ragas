"""
Tests for empty generation exception handling in TestsetGenerator (#1137).

Verifies that when the Executor returns an empty result list (e.g. all jobs
failed silently), TestsetGenerator raises ExceptionInRunner instead of
proceeding with empty data.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

# Mock sknetwork since scikit-network may not be installed in all environments
sys.modules.setdefault("sknetwork", MagicMock())
sys.modules.setdefault("sknetwork.clustering", MagicMock())
sys.modules.setdefault("sknetwork.data", MagicMock())

from ragas.exceptions import ExceptionInRunner
from ragas.testset.graph import KnowledgeGraph
from ragas.testset.persona import Persona
from ragas.testset.synthesizers.generate import TestsetGenerator


def _make_generator() -> TestsetGenerator:
    """Create a TestsetGenerator with mocked LLM and embeddings."""
    llm = MagicMock()
    embedding_model = MagicMock()
    kg = KnowledgeGraph()
    persona = Persona(role="test", goal="test", values=["test"])
    return TestsetGenerator(
        llm=llm,
        embedding_model=embedding_model,
        knowledge_graph=kg,
        persona_list=[persona],
    )


def _make_query_distribution() -> list:
    """Create a minimal query distribution with a mocked scenario."""
    scenario = MagicMock()
    scenario.name = "test_scenario"
    return [(scenario, 1.0)]


@patch("ragas.testset.synthesizers.generate.Executor")
def test_empty_scenario_results_raises_exception(mock_executor_cls):
    """
    When scenario generation returns an empty list, ExceptionInRunner is raised.
    """
    mock_executor = MagicMock()
    mock_executor.results.return_value = []
    mock_executor_cls.return_value = mock_executor

    generator = _make_generator()
    query_distribution = _make_query_distribution()

    with pytest.raises(ExceptionInRunner):
        generator.generate(
            testset_size=1,
            query_distribution=query_distribution,
        )


@patch("ragas.testset.synthesizers.generate.Executor")
def test_empty_sample_results_raises_exception(mock_executor_cls):
    """
    When sample generation returns an empty list, ExceptionInRunner is raised.
    """
    # First Executor (scenarios) returns a valid result;
    # second Executor (samples) returns empty.
    scenario_executor = MagicMock()
    scenario_executor.results.return_value = [[MagicMock()]]

    sample_executor = MagicMock()
    sample_executor.results.return_value = []

    mock_executor_cls.side_effect = [scenario_executor, sample_executor]

    generator = _make_generator()
    query_distribution = _make_query_distribution()

    with pytest.raises(ExceptionInRunner):
        generator.generate(
            testset_size=1,
            query_distribution=query_distribution,
        )

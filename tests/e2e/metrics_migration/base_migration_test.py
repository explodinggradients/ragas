"""Base test class for metrics migration E2E tests."""

from typing import Any, Callable, Dict, List, Optional

import pytest

from .test_utils import (
    assert_score_types,
    compare_scores_with_tolerance,
    create_legacy_sample,
    print_score_comparison,
    print_test_header,
    print_test_success,
)


class BaseMigrationTest:
    """Base class for metrics migration E2E tests.

    Provides common functionality for testing compatibility between legacy and v2 implementations.
    Subclasses should implement metric-specific test data and configurations.
    """

    @pytest.mark.asyncio
    async def run_e2e_compatibility_test(
        self,
        sample_data: List[Dict[str, Any]],
        legacy_metric_factory: Callable,
        v2_metric_factory: Callable,
        v2_score_method_name: str = "ascore",
        legacy_components: Optional[Dict[str, Any]] = None,
        v2_components: Optional[Dict[str, Any]] = None,
        tolerance: float = 0.3,
        metric_name: str = "Metric",
        additional_info_keys: Optional[List[str]] = None,
    ) -> None:
        """Run E2E compatibility test between legacy and v2 implementations.

        Args:
            sample_data: List of test cases, each as a dictionary
            legacy_metric_factory: Function to create legacy metric instance
            v2_metric_factory: Function to create v2 metric instance
            v2_score_method_name: Name of the scoring method on v2 metric
            legacy_components: Components for legacy metric (llm, embeddings, etc.)
            v2_components: Components for v2 metric (llm, embeddings, etc.)
            tolerance: Maximum allowed score difference
            metric_name: Name of the metric for display
            additional_info_keys: Keys from data dict to display in test output
        """
        # Check if required components are available
        if legacy_components:
            if any(component is None for component in legacy_components.values()):
                pytest.skip("Required components not available for E2E testing")

        if v2_components:
            if any(component is None for component in v2_components.values()):
                pytest.skip("Required components not available for E2E testing")

        # Create metric instances
        legacy_metric = (
            legacy_metric_factory(**legacy_components)
            if legacy_components
            else legacy_metric_factory()
        )
        v2_metric = (
            v2_metric_factory(**v2_components) if v2_components else v2_metric_factory()
        )

        # Run tests for each sample
        for i, data in enumerate(sample_data):
            description = data.get("description", "No description")

            # Prepare additional info for display
            additional_info = {}
            if additional_info_keys:
                for key in additional_info_keys:
                    if key in data:
                        additional_info[key.replace("_", " ").title()] = str(data[key])

            print_test_header(metric_name, i + 1, description, additional_info)

            # Score with legacy implementation
            legacy_sample = create_legacy_sample(data)
            legacy_score = await legacy_metric._single_turn_ascore(legacy_sample, None)

            # Score with v2 implementation
            # Extract parameters for v2 scoring (exclude metadata keys)
            v2_params = {k: v for k, v in data.items() if k != "description"}
            v2_score_method = getattr(v2_metric, v2_score_method_name)
            v2_result = await v2_score_method(**v2_params)

            # Compare scores
            print_score_comparison(legacy_score, v2_result.value)

            # Assert scores are within tolerance
            compare_scores_with_tolerance(
                legacy_score,
                v2_result.value,
                tolerance,
                description,
                i + 1,
            )

            # Assert types and ranges
            assert_score_types(legacy_score, v2_result)

            print_test_success()

    @pytest.mark.asyncio
    async def run_metric_specific_test(
        self,
        test_cases: List[Dict[str, Any]],
        legacy_metric_factory: Callable,
        v2_metric_factory: Callable,
        legacy_components: Optional[Dict[str, Any]] = None,
        v2_components: Optional[Dict[str, Any]] = None,
        test_name: str = "Metric Specific Test",
        assertion_fn: Optional[Callable] = None,
    ) -> None:
        """Run a metric-specific test with custom assertions.

        Args:
            test_cases: List of test cases
            legacy_metric_factory: Function to create legacy metric instance
            v2_metric_factory: Function to create v2 metric instance
            legacy_components: Components for legacy metric
            v2_components: Components for v2 metric
            test_name: Name of the test for display
            assertion_fn: Optional custom assertion function that takes (case, legacy_score, v2_result)
        """
        # Check if required components are available
        if legacy_components:
            if any(component is None for component in legacy_components.values()):
                pytest.skip("Required components not available for testing")

        if v2_components:
            if any(component is None for component in v2_components.values()):
                pytest.skip("Required components not available for testing")

        # Create metric instances
        legacy_metric = (
            legacy_metric_factory(**legacy_components)
            if legacy_components
            else legacy_metric_factory()
        )
        v2_metric = (
            v2_metric_factory(**v2_components) if v2_components else v2_metric_factory()
        )

        # Run tests for each case
        for case in test_cases:
            description = case.get("description", "No description")
            print(f"\n🎯 Testing {test_name}: {description}")

            # Score with legacy implementation
            legacy_sample = create_legacy_sample(case)
            legacy_score = await legacy_metric._single_turn_ascore(legacy_sample, None)

            # Score with v2 implementation
            v2_params = {
                k: v
                for k, v in case.items()
                if k not in ["description", "expected_high", "expected_low"]
            }
            v2_result = await v2_metric.ascore(**v2_params)

            # Print scores
            print_score_comparison(legacy_score, v2_result.value)

            # Run custom assertions if provided
            if assertion_fn:
                assertion_fn(case, legacy_score, v2_result)
            else:
                # Default: just verify types
                assert_score_types(legacy_score, v2_result)

    def create_requirements_documentation(
        self,
        metric_name: str,
        requirements: Dict[str, str],
        test_file_name: str,
    ) -> None:
        """Print documentation about E2E test requirements.

        Args:
            metric_name: Name of the metric
            requirements: Dictionary of requirements
            test_file_name: Name of the test file
        """
        print(f"\n📋 {metric_name} E2E Test Requirements:")
        for key, value in requirements.items():
            print(f"   {key.capitalize()}: {value}")

        print("\n🚀 To enable full E2E testing:")
        print("   1. Configure required providers (e.g., export OPENAI_API_KEY=...)")
        print("   2. Remove @pytest.mark.skip decorators")
        print(f"   3. Run: pytest tests/e2e/metrics_migration/{test_file_name} -v -s")

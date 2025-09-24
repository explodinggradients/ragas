import asyncio
import warnings
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestAsyncUtilsControl:
    """Test nest_asyncio application control."""

    def test_run_with_nest_asyncio_default(self):
        """Test run function applies nest_asyncio by default."""
        from ragas.async_utils import run

        async def test_func():
            return "test"

        with patch("ragas.async_utils.apply_nest_asyncio") as mock_apply:
            result = run(test_func)

        mock_apply.assert_called_once()
        assert result == "test"

    def test_run_without_nest_asyncio(self):
        """Test run function can skip nest_asyncio."""
        from ragas.async_utils import run

        async def test_func():
            return "test"

        with patch("ragas.async_utils.apply_nest_asyncio") as mock_apply:
            result = run(test_func, allow_nest_asyncio=False)

        mock_apply.assert_not_called()
        assert result == "test"


class TestEvaluateAsyncControl:
    """Test the sync evaluate function with async options."""

    def test_evaluate_with_nest_asyncio_default(self):
        """Test evaluate with default nest_asyncio behavior."""
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
                message=".*coroutine.*was never awaited",
            )

            with patch("ragas.async_utils.run") as mock_run:
                mock_run.return_value = MagicMock()

                from ragas import evaluate

                evaluate(
                    dataset=MagicMock(),
                    metrics=[MagicMock()],
                    show_progress=False,
                )

        # Should call run() which applies nest_asyncio by default
        mock_run.assert_called_once()

    def test_evaluate_allow_nest_asyncio_true(self):
        """Test evaluate with allow_nest_asyncio=True explicitly."""
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
                message=".*coroutine.*was never awaited",
            )

            with patch("ragas.async_utils.run") as mock_run:
                mock_run.return_value = MagicMock()

                from ragas import evaluate

                evaluate(
                    dataset=MagicMock(),
                    metrics=[MagicMock()],
                    show_progress=False,
                    allow_nest_asyncio=True,
                )

        # Should use run() which applies nest_asyncio
        mock_run.assert_called_once()

    def test_evaluate_allow_nest_asyncio_false(self):
        """Test evaluate with allow_nest_asyncio=False."""
        with warnings.catch_warnings():
            # Suppress RuntimeWarning about unawaited coroutines in tests
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
                message=".*coroutine.*was never awaited",
            )

            with patch("asyncio.run") as mock_asyncio_run:
                with patch("ragas.async_utils.run") as mock_run:
                    mock_asyncio_run.return_value = MagicMock()

                    from ragas import evaluate

                    evaluate(
                        dataset=MagicMock(),
                        metrics=[MagicMock()],
                        show_progress=False,
                        allow_nest_asyncio=False,
                    )

        # Should use asyncio.run, not ragas.async_utils.run
        mock_asyncio_run.assert_called_once()
        mock_run.assert_not_called()


class TestAevaluateImport:
    """Test that aevaluate can be imported and is async."""

    def test_aevaluate_importable(self):
        """Test that aevaluate can be imported."""
        from ragas import aevaluate

        assert callable(aevaluate)
        assert asyncio.iscoroutinefunction(aevaluate)

    def test_evaluate_has_allow_nest_asyncio_param(self):
        """Test that evaluate function has the new parameter."""
        import inspect

        from ragas import evaluate

        sig = inspect.signature(evaluate)
        assert "allow_nest_asyncio" in sig.parameters
        assert sig.parameters["allow_nest_asyncio"].default is True


class TestNestAsyncioNotAppliedInAevaluate:
    """Test that aevaluate doesn't apply nest_asyncio."""

    @pytest.mark.asyncio
    async def test_aevaluate_no_nest_asyncio_applied(self):
        """Test that aevaluate doesn't call apply_nest_asyncio."""
        with warnings.catch_warnings():
            # Suppress RuntimeWarning about unawaited coroutines in tests
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
                message=".*coroutine.*was never awaited",
            )

            # Mock all the dependencies to avoid actual API calls
            with patch("ragas.evaluation.EvaluationDataset"):
                with patch("ragas.evaluation.validate_required_columns"):
                    with patch("ragas.evaluation.validate_supported_metrics"):
                        with patch("ragas.evaluation.Executor") as mock_executor_class:
                            with patch("ragas.evaluation.new_group"):
                                with patch(
                                    "ragas.async_utils.apply_nest_asyncio"
                                ) as mock_apply:
                                    # Mock executor
                                    mock_executor = MagicMock()
                                    mock_executor.aresults = AsyncMock(
                                        return_value=[0.8]
                                    )
                                    mock_executor_class.return_value = mock_executor

                                    # Mock dataset
                                    mock_dataset_instance = MagicMock()
                                    mock_dataset_instance.get_sample_type.return_value = MagicMock()
                                    mock_dataset_instance.__iter__ = lambda x: iter([])

                                    from ragas import aevaluate

                                    try:
                                        await aevaluate(
                                            dataset=mock_dataset_instance,
                                            metrics=[],
                                            show_progress=False,
                                        )
                                    except Exception:
                                        pass

            # aevaluate should never call apply_nest_asyncio
            mock_apply.assert_not_called()


class TestAsyncIntegration:
    """Basic integration tests for async scenarios."""

    @pytest.mark.asyncio
    async def test_aevaluate_in_running_loop(self):
        """Test aevaluate can be called when an event loop is already running."""
        # This test runs with pytest-asyncio, so an event loop is running
        from ragas import aevaluate

        # Just test that the function can be called without RuntimeError
        # We'll mock everything to avoid API calls
        with patch("ragas.evaluation.EvaluationDataset"):
            with patch("ragas.evaluation.validate_required_columns"):
                with patch("ragas.evaluation.validate_supported_metrics"):
                    with patch("ragas.evaluation.Executor") as mock_executor_class:
                        with patch("ragas.evaluation.new_group"):
                            mock_executor = MagicMock()
                            mock_executor.aresults = AsyncMock(return_value=[])
                            mock_executor_class.return_value = mock_executor

                            try:
                                await aevaluate(
                                    dataset=MagicMock(),
                                    metrics=[],
                                    show_progress=False,
                                )
                                # Should not raise RuntimeError about event loop
                            except Exception as e:
                                # We expect other exceptions due to mocking, but not RuntimeError
                                assert "event loop" not in str(e).lower()
                                assert "nest_asyncio" not in str(e).lower()

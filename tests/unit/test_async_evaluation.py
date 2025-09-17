"""
Test async evaluation functionality including aevaluate and improved evaluate.
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from ragas import aevaluate, evaluate
from ragas.async_utils import run_sync_from_async
from ragas.dataset_schema import EvaluationDataset


class TestAsyncEvaluation:
    """Test cases for async evaluation functionality."""

    def test_aevaluate_import(self):
        """Test that aevaluate can be imported from ragas."""
        from ragas import aevaluate

        assert callable(aevaluate)

    @pytest.mark.asyncio
    async def test_aevaluate_no_nest_asyncio(self):
        """Test that aevaluate doesn't call nest_asyncio.apply()."""
        empty_dataset = EvaluationDataset.from_list([])

        with patch("nest_asyncio.apply") as mock_apply:
            try:
                # This will likely fail due to empty dataset, but that's ok
                _ = await aevaluate(empty_dataset, metrics=[])
            except Exception:
                pass  # Expected to fail with empty dataset

            # The key test: nest_asyncio.apply() should NOT be called
            mock_apply.assert_not_called()

    @pytest.mark.asyncio
    async def test_aevaluate_in_event_loop(self):
        """Test that aevaluate works when called from within an event loop."""
        from ragas.async_utils import is_event_loop_running

        # Verify we're in an event loop
        assert is_event_loop_running() is True

        empty_dataset = EvaluationDataset.from_list([])

        # This should not raise "asyncio.run() cannot be called from a running event loop"
        with patch("nest_asyncio.apply") as mock_apply:
            try:
                result = await aevaluate(empty_dataset, metrics=[])
                assert result is not None  # If it succeeds
            except Exception as e:
                # Failure is expected with empty dataset, but should not be event loop error
                assert (
                    "cannot be called from a running event loop" not in str(e).lower()
                )

            # nest_asyncio should not be called
            mock_apply.assert_not_called()


class TestRunSyncFromAsync:
    """Test the run_sync_from_async utility function."""

    def test_run_sync_from_async_no_loop(self):
        """Test run_sync_from_async when no event loop is running."""

        async def test_coro():
            return "test_result"

        result = run_sync_from_async(test_coro)
        assert result == "test_result"

    def test_run_sync_from_async_with_loop_jupyter(self):
        """Test run_sync_from_async in Jupyter-like environment."""

        async def test_coro():
            return "jupyter_result"

        # Mock IPython to simulate Jupyter environment
        with patch("IPython.get_ipython") as mock_get_ipython:
            mock_get_ipython.return_value = MagicMock()  # Simulates Jupyter

            # Mock is_event_loop_running to return True
            with patch("ragas.async_utils.is_event_loop_running", return_value=True):
                with patch("nest_asyncio.apply") as mock_apply:
                    with patch(
                        "asyncio.run", return_value="jupyter_result"
                    ) as mock_run:
                        result = run_sync_from_async(test_coro)

                        assert result == "jupyter_result"
                        mock_apply.assert_called_once()  # Should apply nest_asyncio in Jupyter
                        mock_run.assert_called_once()

    def test_run_sync_from_async_server_context_error(self):
        """Test run_sync_from_async raises helpful error in server contexts."""

        async def test_coro():
            return "server_result"

        # Mock event loop running but no IPython (server context)
        with patch("ragas.async_utils.is_event_loop_running", return_value=True):
            with patch("IPython.get_ipython", side_effect=ImportError):
                with pytest.raises(RuntimeError) as exc_info:
                    run_sync_from_async(test_coro)

                error_msg = str(exc_info.value)
                assert "Use 'aevaluate()' instead" in error_msg
                assert "FastAPI servers" in error_msg


class TestEvaluateAsyncImprovement:
    """Test the improved evaluate function that uses aevaluate internally."""

    def test_evaluate_sync_context(self):
        """Test evaluate() in normal sync context."""
        empty_dataset = EvaluationDataset.from_list([])

        with patch("ragas.async_utils.is_event_loop_running", return_value=False):
            with patch("asyncio.run") as mock_run:
                mock_run.return_value = MagicMock()  # Mock result

                try:
                    _ = evaluate(empty_dataset, metrics=[])
                    # If it doesn't raise, asyncio.run was called
                    mock_run.assert_called_once()
                except Exception:
                    # Even if aevaluate fails, asyncio.run should have been called
                    mock_run.assert_called_once()

    def test_evaluate_return_executor_fallback(self):
        """Test evaluate() with return_executor=True uses old implementation."""
        empty_dataset = EvaluationDataset.from_list([])

        # This should not call the new async path
        with patch("ragas.async_utils.run_sync_from_async") as mock_run_sync:
            try:
                result = evaluate(empty_dataset, metrics=[], return_executor=True)
                # Should not call the new async implementation
                mock_run_sync.assert_not_called()

                # Should return an Executor
                from ragas.executor import Executor

                assert isinstance(result, Executor)
            except Exception:
                # Even if it fails, should not have called new async path
                mock_run_sync.assert_not_called()

    def test_evaluate_async_context_error(self):
        """Test evaluate() gives helpful error in async contexts."""

        async def test_in_async():
            empty_dataset = EvaluationDataset.from_list([])

            with pytest.raises(RuntimeError) as exc_info:
                evaluate(empty_dataset, metrics=[])

            error_msg = str(exc_info.value)
            assert "Use 'aevaluate()' instead" in error_msg

        asyncio.run(test_in_async())


@pytest.mark.asyncio
async def test_aevaluate_signature_compatibility():
    """Test that aevaluate has the same signature as evaluate (minus return_executor)."""
    import inspect

    # Get function signatures
    evaluate_sig = inspect.signature(evaluate)
    aevaluate_sig = inspect.signature(aevaluate)

    # aevaluate should have all parameters except return_executor
    evaluate_params = set(evaluate_sig.parameters.keys())
    aevaluate_params = set(aevaluate_sig.parameters.keys())

    # return_executor should be in evaluate but not aevaluate
    assert "return_executor" in evaluate_params
    assert "return_executor" not in aevaluate_params

    # All other parameters should be the same
    evaluate_params.remove("return_executor")
    assert evaluate_params == aevaluate_params

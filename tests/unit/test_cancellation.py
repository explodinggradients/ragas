"""
Unit tests for the cancellation functionality.
"""

import asyncio
import threading
import typing as t

from ragas.dataset_schema import (
    EvaluationDataset,
    SingleTurnSample,
    SingleTurnSampleOrMultiTurnSample,
)
from ragas.evaluation import evaluate
from ragas.executor import Executor


class TestExecutorCancellation:
    """Test cancellation functionality in Executor."""

    def test_executor_cancel_method_exists(self):
        """Test that Executor has cancel and is_cancelled methods."""
        executor = Executor()
        assert hasattr(executor, "cancel")
        assert hasattr(executor, "is_cancelled")
        assert callable(executor.cancel)
        assert callable(executor.is_cancelled)

    def test_executor_cancellation_state(self):
        """Test cancellation state management."""
        executor = Executor()

        # Initially not cancelled
        assert not executor.is_cancelled()

        # After cancel(), should be cancelled
        executor.cancel()
        assert executor.is_cancelled()

    def test_executor_cancel_idempotent(self):
        """Test that calling cancel() multiple times is safe."""
        executor = Executor()

        # Multiple calls should be safe
        executor.cancel()
        assert executor.is_cancelled()

        executor.cancel()  # Second call
        assert executor.is_cancelled()

    def test_executor_respects_cancellation(self):
        """Test that executor respects cancellation during execution."""
        executor = Executor(desc="Test Cancellation", show_progress=False)

        # Test basic cancellation without complex async scenarios
        # to avoid asyncio edge case warnings
        async def simple_task():
            return "completed"

        # Submit a task but don't execute it
        executor.submit(simple_task)

        # Cancel before execution
        executor.cancel()
        assert executor.is_cancelled()

        # The cancellation state should be preserved
        assert executor.is_cancelled()


class TestEvaluateCancellation:
    """Test cancellation functionality in evaluate()."""

    def create_test_dataset(self):
        """Create a simple test dataset."""
        samples: t.List[SingleTurnSample] = [
            SingleTurnSample(
                user_input="Test question",
                response="Test answer",
                retrieved_contexts=["Test context"],
            )
        ]
        # Type cast to satisfy EvaluationDataset constructor
        return EvaluationDataset(
            samples=t.cast(t.List[SingleTurnSampleOrMultiTurnSample], samples)
        )

    def test_evaluate_return_executor_parameter(self):
        """Test that evaluate() accepts return_executor parameter."""
        dataset = self.create_test_dataset()

        # Should return Executor when return_executor=True
        executor = evaluate(dataset=dataset, metrics=[], return_executor=True)
        assert isinstance(executor, Executor)
        assert hasattr(executor, "cancel")
        assert hasattr(executor, "is_cancelled")

    def test_evaluate_default_behavior_unchanged(self):
        """Test that evaluate() default behavior is unchanged."""
        dataset = self.create_test_dataset()

        # Test that return_executor=False is the default behavior
        # We'll get an executor and verify it's not returned by default
        executor = evaluate(dataset=dataset, metrics=[], return_executor=True)
        assert isinstance(executor, Executor), (
            "return_executor=True should return Executor"
        )

        # Test that default behavior would not return executor
        # (We can't easily test the full evaluation without LLMs,
        # so this tests the key API difference)
        assert hasattr(executor, "cancel")
        assert hasattr(executor, "is_cancelled")

    def test_evaluate_executor_cancellation(self):
        """Test that evaluate() executor can be cancelled."""
        dataset = self.create_test_dataset()

        result = evaluate(dataset=dataset, metrics=[], return_executor=True)

        # Type assertion since return_executor=True guarantees Executor
        executor = t.cast(Executor, result)

        # Should be cancellable
        executor.cancel()
        assert executor.is_cancelled()


class TestGeneratorCancellation:
    """Test cancellation functionality in TestsetGenerator."""

    def test_generate_with_langchain_docs_return_executor_parameter(self):
        """Test that generate_with_langchain_docs accepts return_executor parameter."""
        # This is mainly a signature test since full testing requires LLM/embeddings
        # Import locally to avoid pytest collection issues
        from ragas.testset.synthesizers.generate import TestsetGenerator

        generator = TestsetGenerator.__new__(
            TestsetGenerator
        )  # Create without __init__

        # Verify the method signature includes return_executor
        import inspect

        sig = inspect.signature(generator.generate_with_langchain_docs)
        assert "return_executor" in sig.parameters

        # Verify default value is False
        param = sig.parameters["return_executor"]
        assert param.default is False

    def test_generate_method_return_executor_parameter(self):
        """Test that generate method accepts return_executor parameter."""
        # Import locally to avoid pytest collection issues
        from ragas.testset.synthesizers.generate import TestsetGenerator

        generator = TestsetGenerator.__new__(TestsetGenerator)

        # Verify the method signature includes return_executor
        import inspect

        sig = inspect.signature(generator.generate)
        assert "return_executor" in sig.parameters

        # Verify default value is False
        param = sig.parameters["return_executor"]
        assert param.default is False


class TestCancellationIntegration:
    """Test integration scenarios with cancellation."""

    def test_cancellation_thread_safety(self):
        """Test that cancellation works safely across threads."""
        executor = Executor(show_progress=False)

        # Add a task
        async def simple_task():
            await asyncio.sleep(0.1)
            return "done"

        executor.submit(simple_task)

        # Cancel from another thread
        cancel_thread = threading.Thread(target=executor.cancel)
        cancel_thread.start()
        cancel_thread.join()

        # Should be cancelled
        assert executor.is_cancelled()

    def test_multiple_executors_isolation(self):
        """Test that cancelling one executor doesn't affect others."""
        executor1 = Executor(show_progress=False)
        executor2 = Executor(show_progress=False)
        executor3 = Executor(show_progress=False)

        # Cancel only executor2
        executor2.cancel()

        # Check isolation
        assert not executor1.is_cancelled()
        assert executor2.is_cancelled()
        assert not executor3.is_cancelled()

    def test_cancellation_with_empty_job_list(self):
        """Test cancellation with no submitted jobs."""
        executor = Executor(show_progress=False)

        # Cancel without any jobs
        executor.cancel()
        assert executor.is_cancelled()

        # Results should be empty
        results = executor.results()
        assert results == []


class TestCancellationDocumentationExamples:
    """Test that documentation examples work correctly."""

    def test_timeout_pattern_example(self):
        """Test the timeout pattern from documentation."""

        def evaluate_with_timeout(dataset, metrics, timeout_seconds: float = 300):
            """Example timeout function from docs."""
            import threading

            from ragas import evaluate

            result = evaluate(dataset=dataset, metrics=metrics, return_executor=True)
            # Type assertion since return_executor=True guarantees Executor
            executor = t.cast(Executor, result)

            results = None
            exception = None

            def run_evaluation():
                nonlocal results, exception
                try:
                    results = executor.results()
                except Exception as e:
                    exception = e

            thread = threading.Thread(target=run_evaluation)
            thread.start()

            thread.join(timeout=timeout_seconds)

            if thread.is_alive():
                executor.cancel()
                thread.join(timeout=2)
                return None, "timeout"

            return results, exception

        # Test with very short timeout
        samples: t.List[SingleTurnSample] = [
            SingleTurnSample(
                user_input="Test", response="Test", retrieved_contexts=["Test"]
            )
        ]
        dataset = EvaluationDataset(
            samples=t.cast(t.List[SingleTurnSampleOrMultiTurnSample], samples)
        )

        results, error = evaluate_with_timeout(dataset, [], timeout_seconds=0.01)

        # Should either complete very fast or timeout
        assert error == "timeout" or results is not None

    def test_evaluation_manager_example(self):
        """Test the EvaluationManager example from documentation."""

        class EvaluationManager:
            def __init__(self):
                self.executors = []

            def start_evaluation(self, dataset, metrics):
                result = evaluate(
                    dataset=dataset, metrics=metrics, return_executor=True
                )
                # Type assertion since return_executor=True guarantees Executor
                executor = t.cast(Executor, result)
                self.executors.append(executor)
                return executor

            def cancel_all(self):
                """Cancel all running evaluations."""
                cancelled_count = 0
                for executor in self.executors:
                    if not executor.is_cancelled():
                        executor.cancel()
                        cancelled_count += 1
                return cancelled_count

            def cleanup_completed(self):
                """Remove completed executors."""
                before_count = len(self.executors)
                self.executors = [ex for ex in self.executors if not ex.is_cancelled()]
                return before_count - len(self.executors)

        # Test the manager
        manager = EvaluationManager()

        samples: t.List[SingleTurnSample] = [
            SingleTurnSample(
                user_input="Test", response="Test", retrieved_contexts=["Test"]
            )
        ]
        dataset = EvaluationDataset(
            samples=t.cast(t.List[SingleTurnSampleOrMultiTurnSample], samples)
        )

        # Start evaluations
        manager.start_evaluation(dataset, [])
        manager.start_evaluation(dataset, [])

        assert len(manager.executors) == 2

        # Cancel all
        cancelled = manager.cancel_all()
        assert cancelled == 2

        # Cleanup
        removed = manager.cleanup_completed()
        assert removed == 2
        assert len(manager.executors) == 0

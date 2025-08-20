"""
Simple test to validate tracing integration works.
"""

import pytest


def test_basic_tracing_import():
    """Test that basic imports work."""
    try:
        from ragas.integrations.tracing import observe

        assert callable(observe)
        print("✓ Import successful")
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")


def test_observe_decorator():
    """Test the observe decorator works as no-op."""
    from ragas.integrations.tracing import observe

    @observe()  # type: ignore
    def test_function():
        return "success"

    result = test_function()
    assert result == "success"
    print("✓ Decorator works")


def test_callback_compatibility():
    """Test that tracing doesn't interfere with existing callbacks."""
    from ragas.callbacks import RagasTracer
    from ragas.integrations.tracing import observe

    tracer = RagasTracer()

    @observe()  # type: ignore
    def traced_function():
        return {"metric": "value"}

    result = traced_function()
    assert result["metric"] == "value"

    # Tracer should still be functional
    assert isinstance(tracer.traces, dict)
    print("✓ Callback compatibility works")


def test_no_experimental_imports():
    """Test that experimental imports are no longer available."""
    try:
        # Try importing from the removed experimental path
        import importlib.util

        spec = importlib.util.find_spec("ragas.experimental.tracing.langfuse")
        assert spec is None, "Experimental module should not be available"
    except ImportError:
        pass  # Expected behavior
    print("✓ Experimental imports correctly removed")


if __name__ == "__main__":
    test_basic_tracing_import()
    test_observe_decorator()
    test_callback_compatibility()
    test_no_experimental_imports()
    print("All tests passed!")

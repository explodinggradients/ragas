"""
Comprehensive test suite for tracing integrations.

Tests both Langfuse and MLflow integrations with proper mocking
to avoid external dependencies in tests.
"""

import asyncio
import os
import sys
import typing as t
from datetime import datetime
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest


class TestLangfuseIntegration:
    """Test suite for Langfuse tracing integration."""

    def test_langfuse_imports_with_missing_dependency(self):
        """Test that imports work gracefully when langfuse is not available."""
        with patch.dict("sys.modules", {"langfuse": None, "langfuse.api": None}):
            # This should not raise an ImportError
            from ragas.integrations.tracing.langfuse import (
                LangfuseTrace,
                observe,
                sync_trace,
            )

            assert callable(observe)
            assert LangfuseTrace is not None
            assert callable(sync_trace)

    def test_langfuse_imports_with_dependency_available(self):
        """Test imports when langfuse is available."""
        # Mock langfuse modules
        mock_langfuse = MagicMock()
        mock_api = MagicMock()
        
        with patch.dict("sys.modules", {"langfuse": mock_langfuse, "langfuse.api": mock_api}):
            from ragas.integrations.tracing.langfuse import LangfuseTrace, observe
            
            assert LangfuseTrace is not None
            assert callable(observe)

    def test_observe_decorator_fallback(self):
        """Test that observe decorator works as a no-op when langfuse unavailable."""
        with patch.dict("sys.modules", {"langfuse": None}):
            from ragas.integrations.tracing.langfuse import observe

            @observe()
            def test_function():
                return "test_result"

            result = test_function()
            assert result == "test_result"

    def test_langfuse_trace_initialization(self):
        """Test LangfuseTrace initialization with mock trace."""
        from ragas.integrations.tracing.langfuse import LangfuseTrace, TraceWithFullDetails

        mock_trace = TraceWithFullDetails(
            id="test-trace-id",
            timestamp=datetime.now(),
            htmlPath="test-path",
            latency=100,
            totalCost=0.01,
        )
        
        langfuse_trace = LangfuseTrace(mock_trace)
        assert langfuse_trace.trace == mock_trace

    @pytest.mark.asyncio
    async def test_sync_trace_with_trace_id(self):
        """Test sync_trace function with explicit trace ID."""
        from ragas.integrations.tracing.langfuse import sync_trace

        # Mock the Langfuse client
        with patch("ragas.integrations.tracing.langfuse.Langfuse") as mock_langfuse_class:
            mock_client = MagicMock()
            mock_langfuse_class.return_value = mock_client
            
            result = await sync_trace(trace_id="test-trace-id", max_retries=1, delay=0.1)
            
            assert result is not None
            assert hasattr(result, 'trace')

    @pytest.mark.asyncio
    async def test_sync_trace_without_trace_id(self):
        """Test sync_trace function without trace ID (uses current trace)."""
        from ragas.integrations.tracing.langfuse import sync_trace

        with patch("ragas.integrations.tracing.langfuse.Langfuse") as mock_langfuse_class:
            mock_client = MagicMock()
            mock_client.get_current_trace_id.return_value = "current-trace-id"
            mock_langfuse_class.return_value = mock_client
            
            result = await sync_trace(max_retries=1, delay=0.1)
            
            assert result is not None
            mock_client.get_current_trace_id.assert_called_once()

    @pytest.mark.asyncio
    async def test_sync_trace_no_trace_found(self):
        """Test sync_trace raises ValueError when no trace is found."""
        from ragas.integrations.tracing.langfuse import sync_trace

        with patch("ragas.integrations.tracing.langfuse.Langfuse") as mock_langfuse_class:
            mock_client = MagicMock()
            mock_client.get_current_trace_id.return_value = None
            mock_langfuse_class.return_value = mock_client
            
            with pytest.raises(ValueError, match="No trace id found"):
                await sync_trace(max_retries=1, delay=0.1)

    def test_add_query_param(self):
        """Test URL query parameter addition utility."""
        from ragas.integrations.tracing.langfuse import add_query_param

        base_url = "https://example.com/trace"
        result = add_query_param(base_url, "param", "value")
        
        assert "param=value" in result
        assert result.startswith("https://example.com/trace")

    def test_add_query_param_existing_params(self):
        """Test URL query parameter addition with existing parameters."""
        from ragas.integrations.tracing.langfuse import add_query_param

        base_url = "https://example.com/trace?existing=param"
        result = add_query_param(base_url, "new", "value")
        
        assert "existing=param" in result
        assert "new=value" in result


class TestMLflowIntegration:
    """Test suite for MLflow tracing integration."""

    def test_mlflow_imports_with_missing_dependency(self):
        """Test that imports work gracefully when mlflow is not available."""
        with patch.dict("sys.modules", {"mlflow": None, "mlflow.entities": None}):
            from ragas.integrations.tracing.mlflow import MLflowTrace, sync_trace

            assert MLflowTrace is not None
            assert callable(sync_trace)

    def test_mlflow_imports_with_dependency_available(self):
        """Test imports when mlflow is available."""
        mock_mlflow = MagicMock()
        mock_entities = MagicMock()
        
        with patch.dict("sys.modules", {"mlflow": mock_mlflow, "mlflow.entities": mock_entities}):
            from ragas.integrations.tracing.mlflow import MLflowTrace
            
            assert MLflowTrace is not None

    def test_mlflow_trace_initialization(self):
        """Test MLflowTrace initialization with mock trace."""
        from ragas.integrations.tracing.mlflow import MLflowTrace, Trace

        mock_trace = Trace()
        mlflow_trace = MLflowTrace(mock_trace)
        assert mlflow_trace.trace == mock_trace

    def test_mlflow_trace_get_url_with_env(self):
        """Test MLflowTrace URL generation with MLFLOW_HOST set."""
        from ragas.integrations.tracing.mlflow import MLflowTrace, Trace

        mock_trace = Trace()
        mock_trace.info = MagicMock()
        mock_trace.info.request_id = "test-request-id"
        mock_trace.info.experiment_id = "test-experiment-id"
        
        with patch.dict(os.environ, {"MLFLOW_HOST": "https://mlflow.example.com/"}):
            mlflow_trace = MLflowTrace(mock_trace)
            url = mlflow_trace.get_url()
            
            assert "https://mlflow.example.com" in url
            assert "test-request-id" in url
            assert "test-experiment-id" in url

    def test_mlflow_trace_get_url_no_env(self):
        """Test MLflowTrace URL generation without MLFLOW_HOST."""
        from ragas.integrations.tracing.mlflow import MLflowTrace, Trace

        mock_trace = Trace()
        mlflow_trace = MLflowTrace(mock_trace)
        
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="MLFLOW_HOST environment variable is not set"):
                mlflow_trace.get_url()

    def test_mlflow_trace_filter(self):
        """Test MLflowTrace span filtering."""
        from ragas.integrations.tracing.mlflow import MLflowTrace, Span, Trace

        mock_span = Span()
        mock_span.name = "test-span"
        
        mock_trace = Trace()
        mock_trace.search_spans = MagicMock(return_value=[mock_span])
        
        mlflow_trace = MLflowTrace(mock_trace)
        filtered_spans = mlflow_trace.get_filter("test-span")
        
        assert len(filtered_spans) == 1
        assert filtered_spans[0] == mock_span
        mock_trace.search_spans.assert_called_once_with(name="test-span")

    @pytest.mark.asyncio
    async def test_mlflow_sync_trace_success(self):
        """Test successful MLflow trace synchronization."""
        from ragas.integrations.tracing.mlflow import sync_trace

        with patch("ragas.integrations.tracing.mlflow.get_last_active_trace_id") as mock_get_id, \
             patch("ragas.integrations.tracing.mlflow.get_trace") as mock_get_trace:
            
            mock_get_id.return_value = "test-trace-id"
            mock_trace = MagicMock()
            mock_get_trace.return_value = mock_trace
            
            result = await sync_trace()
            
            assert result is not None
            assert result.trace == mock_trace
            mock_get_id.assert_called_once()
            mock_get_trace.assert_called_once_with("test-trace-id")

    @pytest.mark.asyncio
    async def test_mlflow_sync_trace_no_active_trace(self):
        """Test MLflow sync_trace when no active trace exists."""
        from ragas.integrations.tracing.mlflow import sync_trace

        with patch("ragas.integrations.tracing.mlflow.get_last_active_trace_id") as mock_get_id:
            mock_get_id.return_value = None
            
            with pytest.raises(ValueError, match="No active trace found"):
                await sync_trace()

    @pytest.mark.asyncio
    async def test_mlflow_sync_trace_not_found(self):
        """Test MLflow sync_trace when trace is not found."""
        from ragas.integrations.tracing.mlflow import sync_trace

        with patch("ragas.integrations.tracing.mlflow.get_last_active_trace_id") as mock_get_id, \
             patch("ragas.integrations.tracing.mlflow.get_trace") as mock_get_trace:
            
            mock_get_id.return_value = "test-trace-id"
            mock_get_trace.return_value = None
            
            with pytest.raises(ValueError, match="Trace not found"):
                await sync_trace()


class TestTracingIntegrationInitModule:
    """Test the tracing integration __init__ module."""

    def test_lazy_import_langfuse_functions(self):
        """Test lazy imports for Langfuse functions."""
        from ragas.integrations.tracing import observe, sync_trace, LangfuseTrace
        
        assert callable(observe)
        assert callable(sync_trace)
        assert LangfuseTrace is not None

    def test_lazy_import_mlflow_classes(self):
        """Test lazy imports for MLflow classes."""
        from ragas.integrations.tracing import MLflowTrace
        
        assert MLflowTrace is not None

    def test_invalid_attribute_access(self):
        """Test that accessing non-existent attributes raises AttributeError."""
        import ragas.integrations.tracing as tracing
        
        with pytest.raises(AttributeError, match="has no attribute 'non_existent'"):
            _ = tracing.non_existent


class TestTracingWithCallbackSystem:
    """Test tracing integrations with the existing callback system."""

    def test_tracing_with_ragas_tracer(self):
        """Test that tracing can work alongside RagasTracer."""
        from ragas.callbacks import RagasTracer
        from ragas.integrations.tracing.langfuse import observe

        tracer = RagasTracer()
        
        @observe()
        def traced_function():
            return "test_result"
        
        # Should work without conflicts
        result = traced_function()
        assert result == "test_result"
        
        # Tracer should still be functional
        assert isinstance(tracer.traces, dict)

    def test_callback_manager_compatibility(self):
        """Test compatibility with LangChain callback manager."""
        from langchain_core.callbacks import CallbackManager
        from ragas.callbacks import RagasTracer
        from ragas.integrations.tracing.langfuse import observe

        tracer = RagasTracer()
        callback_manager = CallbackManager([tracer])
        
        @observe()
        def evaluation_function():
            return {"score": 0.85}
        
        result = evaluation_function()
        assert result["score"] == 0.85
        
        # Should not interfere with callback functionality
        assert len(callback_manager.handlers) == 1


if __name__ == "__main__":
    pytest.main([__file__])
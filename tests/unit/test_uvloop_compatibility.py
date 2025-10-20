"""Test uvloop compatibility with nest_asyncio."""

import asyncio
import sys

import pytest


class TestUvloopCompatibility:
    """Test that ragas works with uvloop event loops."""

    @pytest.mark.skipif(sys.version_info < (3, 8), reason="uvloop requires Python 3.8+")
    def test_apply_nest_asyncio_with_uvloop_returns_false(self):
        """Test that apply_nest_asyncio returns False with uvloop."""
        uvloop = pytest.importorskip("uvloop")

        from ragas.async_utils import apply_nest_asyncio

        async def test_func():
            result = apply_nest_asyncio()
            return result

        uvloop.install()
        try:
            result = asyncio.run(test_func())
            assert result is False
        finally:
            asyncio.set_event_loop_policy(None)

    @pytest.mark.skipif(sys.version_info < (3, 8), reason="uvloop requires Python 3.8+")
    def test_run_with_uvloop_and_running_loop(self):
        """Test that run() raises clear error with uvloop in running event loop (Jupyter scenario)."""
        uvloop = pytest.importorskip("uvloop")

        from ragas.async_utils import run

        async def inner_task():
            return "success"

        async def outer_task():
            with pytest.raises(RuntimeError, match="Cannot execute nested async code"):
                run(inner_task)

        uvloop.install()
        try:
            asyncio.run(outer_task())
        finally:
            asyncio.set_event_loop_policy(None)

    @pytest.mark.skipif(sys.version_info < (3, 8), reason="uvloop requires Python 3.8+")
    def test_run_async_tasks_with_uvloop(self):
        """Test that run_async_tasks works with uvloop."""
        uvloop = pytest.importorskip("uvloop")

        from ragas.async_utils import run_async_tasks

        async def task(n):
            return n * 2

        tasks = [task(i) for i in range(5)]

        uvloop.install()
        try:
            results = run_async_tasks(tasks, show_progress=False)
            assert sorted(results) == [0, 2, 4, 6, 8]
        finally:
            asyncio.set_event_loop_policy(None)

    def test_apply_nest_asyncio_without_uvloop_returns_true(self):
        """Test that apply_nest_asyncio returns True with standard asyncio."""
        from ragas.async_utils import apply_nest_asyncio

        async def test_func():
            result = apply_nest_asyncio()
            return result

        result = asyncio.run(test_func())
        assert result is True

    def test_run_with_standard_asyncio_and_running_loop(self):
        """Test that run() works with standard asyncio in a running loop."""
        from ragas.async_utils import run

        async def inner_task():
            return "nested_success"

        async def outer_task():
            result = run(inner_task)
            return result

        result = asyncio.run(outer_task())
        assert result == "nested_success"

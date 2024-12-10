import asyncio
import os

import pytest

from ragas import cacher
from ragas.cache import DiskCacheBackend, _generate_cache_key, _make_hashable


@pytest.fixture(scope="function")
def clear_env():
    """Clear the RAGAS_CACHE_ENABLED environment variable before each test."""
    if "RAGAS_CACHE_ENABLED" in os.environ:
        del os.environ["RAGAS_CACHE_ENABLED"]
    yield
    if "RAGAS_CACHE_ENABLED" in os.environ:
        del os.environ["RAGAS_CACHE_ENABLED"]


@pytest.fixture(scope="function")
def temp_cache_dir(tmp_path):
    """Use a temporary directory for caching."""
    return str(tmp_path)


@pytest.fixture(scope="function")
def cache_backend(temp_cache_dir):
    """Provide a DiskCacheBackend instance with a temporary directory."""
    return DiskCacheBackend(cache_dir=temp_cache_dir)


def test_make_hashable():
    """Test that _make_hashable converts various objects into a hashable structure."""
    data = {"tuple": (1, 2), "list": [3, 4], "set": {5, 6}, "dict": {"a": 1, "b": 2}}

    result = _make_hashable(data)
    # Check that the result is tuple-based (hashable) and stable
    assert isinstance(result, tuple)
    assert len(result) == len(data)


def test_generate_cache_key():
    """Test that cache keys change when arguments or kwargs differ."""

    def sample_func(a, b):
        return a + b

    key1 = _generate_cache_key(sample_func, (1, 2), {})
    key2 = _generate_cache_key(sample_func, (2, 2), {})
    assert key1 != key2, "Cache keys should differ for different args"

    key3 = _generate_cache_key(sample_func, (1, 2), {"c": 3})
    assert key1 != key3, "Cache keys should differ if kwargs differ"


def test_caching_disabled_by_default(clear_env, cache_backend):
    """Test that caching is disabled by default if RAGAS_CACHE_ENABLED is not set."""

    @cacher(cache_backend=cache_backend)
    def add(x, y):
        return x + y

    # Call twice, check that no caching occurred by comparing new results from a function with side effects
    # We'll simulate side effects by returning a changing result.
    @cacher(cache_backend=cache_backend)
    def changing_result():
        import time

        time.sleep(0.01)  # Sleep to ensure a measurable time difference
        return time.time()  # Returns current time, should differ if not cached

    r1 = changing_result()
    r2 = changing_result()
    # If caching was enabled, r1 and r2 would be the same.
    assert r1 != r2, "Should not return cached result when caching is disabled."


def test_caching_enabled(clear_env, cache_backend):
    """Test that caching works when RAGAS_CACHE_ENABLED is set to '1'."""
    os.environ["RAGAS_CACHE_ENABLED"] = "1"

    call_count = {"count": 0}

    @cacher(cache_backend=cache_backend)
    def expensive_function():
        call_count["count"] += 1
        return "expensive_result"

    # First call: should run the function
    result1 = expensive_function()
    assert result1 == "expensive_result"
    assert call_count["count"] == 1

    # Second call: should return cached result, not increment call_count
    result2 = expensive_function()
    assert result2 == "expensive_result"
    assert call_count["count"] == 1, "Call count should not increase on cached result"


@pytest.mark.asyncio
async def test_async_caching_enabled(clear_env, cache_backend):
    """Test that caching works for async functions when enabled."""
    os.environ["RAGAS_CACHE_ENABLED"] = "1"

    call_count = {"count": 0}

    @cacher(cache_backend=cache_backend)
    async def async_expensive_function(x):
        call_count["count"] += 1
        await asyncio.sleep(0.1)
        return x * 2

    # First call
    result1 = await async_expensive_function(10)
    assert result1 == 20
    assert call_count["count"] == 1

    # Second call with same argument should hit cache
    result2 = await async_expensive_function(10)
    assert result2 == 20
    assert call_count["count"] == 1, "Should have come from cache"


def test_caching_with_different_args_enabled(clear_env, cache_backend):
    """Test that different arguments produce different cache entries."""
    os.environ["RAGAS_CACHE_ENABLED"] = "1"

    call_count = {"count": 0}

    @cacher(cache_backend=cache_backend)
    def multiply(x, y):
        call_count["count"] += 1
        return x * y

    assert multiply(2, 3) == 6
    assert multiply(2, 3) == 6
    # Same arguments, should have cached
    assert call_count["count"] == 1

    # Different arguments, cache miss
    assert multiply(3, 3) == 9
    assert call_count["count"] == 2


def test_disable_caching_via_env(clear_env, cache_backend):
    """Test that setting RAGAS_CACHE_ENABLED = '0' disables caching."""
    os.environ["RAGAS_CACHE_ENABLED"] = "0"

    call_count = {"count": 0}

    @cacher(cache_backend=cache_backend)
    def increment():
        call_count["count"] += 1
        return call_count["count"]

    val1 = increment()
    val2 = increment()
    # Caching disabled, should increment each time
    assert val2 == val1 + 1

import asyncio
import os
import tempfile

import pytest

from ragas.utils import (
    async_to_sync,
    batched,
    check_if_sum_is_close,
    create_nano_id,
    get_from_dict,
    get_test_directory,
)


@pytest.mark.parametrize(
    ["values", "close_to", "num_places"],
    [
        [[0.1, 0.2, 0.3], 0.6, 1],
        [[0.8, 0.1, 0.1], 1.0, 1],
        [[0.94, 0.03, 0.03], 1.0, 2],
        [[0.3948, 0.3948, 0.2104], 1.0, 4],
        [[10.19, 10.19, 10.19], 30.57, 2],
    ],
)
def test_check_if_sum_is_close(values, close_to, num_places):
    assert check_if_sum_is_close(values, close_to, num_places)


data_dict = {
    "something": {"nested": {"key": "value"}},
    "other": {"key": "value"},
    "key": "value",
    "another_key": "value",
    "nested_key": {"key": "value"},
}


@pytest.mark.parametrize(
    ["data_dict", "key", "expected"],
    [
        (data_dict, "something.nested.key", "value"),
        (data_dict, "other.key", "value"),
        (data_dict, "something.not_there_in_key", None),
        (data_dict, "something.nested.not_here", None),
    ],
)
def test_get_from_dict(data_dict, key, expected):
    assert get_from_dict(data_dict, key) == expected


@pytest.mark.parametrize(
    ["camel_case_string", "expected"],
    [
        ("myVariableName", "my_variable_name"),
        ("CamelCaseString", "camel_case_string"),
        ("AnotherCamelCaseString", "another_camel_case_string"),
    ],
)
def test_camel_to_snake(camel_case_string, expected):
    from ragas.utils import camel_to_snake

    assert camel_to_snake(camel_case_string) == expected


class TestBatched:
    # Test cases for the `batched` function
    @pytest.mark.parametrize(
        "iterable, n, expected",
        [
            ("ABCDEFG", 3, [("A", "B", "C"), ("D", "E", "F"), ("G",)]),
            ([1, 2, 3, 4, 5, 6, 7], 2, [(1, 2), (3, 4), (5, 6), (7,)]),
            (range(5), 5, [(0, 1, 2, 3, 4)]),
            (["a", "b", "c", "d"], 1, [("a",), ("b",), ("c",), ("d",)]),
            ([], 3, []),  # Edge case: empty iterable
        ],
    )
    def test_batched(self, iterable, n: int, expected):
        result = list(batched(iterable, n))
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_batched_invalid_n(self):
        """Test that `batched` raises ValueError if n < 1."""
        with pytest.raises(ValueError, match="n must be at least one"):
            list(batched("ABCDEFG", 0))  # n = 0 should raise ValueError

    @pytest.mark.parametrize(
        "iterable, n, expected_type",
        [
            ("ABCDEFG", 3, str),
            ([1, 2, 3], 2, int),
            (["x", "y", "z"], 1, str),
        ],
    )
    def test_batched_output_type(self, iterable, n, expected_type: type):
        """Test that items in each batch maintain the original data type."""
        result = list(batched(iterable, n))
        for batch in result:
            assert all(isinstance(item, expected_type) for item in batch)


class TestCreateNanoId:
    """Test cases for the create_nano_id function."""

    def test_create_nano_id_default_size(self):
        """Test that create_nano_id generates IDs of default size (12)."""
        nano_id = create_nano_id()
        assert len(nano_id) == 12
        assert nano_id.isalnum()

    def test_create_nano_id_custom_size(self):
        """Test that create_nano_id respects custom size parameter."""
        for size in [5, 8, 16, 20]:
            nano_id = create_nano_id(size=size)
            assert len(nano_id) == size
            assert nano_id.isalnum()

    def test_create_nano_id_uniqueness(self):
        """Test that create_nano_id generates unique IDs."""
        ids = set()
        for _ in range(100):
            nano_id = create_nano_id()
            assert nano_id not in ids, "Generated duplicate ID"
            ids.add(nano_id)

    def test_create_nano_id_alphanumeric(self):
        """Test that create_nano_id only uses alphanumeric characters."""
        nano_id = create_nano_id(size=50)  # Larger size for better coverage
        for char in nano_id:
            assert char.isalnum(), f"Non-alphanumeric character found: {char}"


class TestAsyncToSync:
    """Test cases for the async_to_sync function."""

    def test_async_to_sync_basic(self):
        """Test basic async to sync conversion."""

        async def async_add(a, b):
            await asyncio.sleep(0.001)  # Small delay to make it truly async
            return a + b

        sync_add = async_to_sync(async_add)
        result = sync_add(3, 4)
        assert result == 7

    def test_async_to_sync_with_kwargs(self):
        """Test async to sync conversion with keyword arguments."""

        async def async_multiply(x, multiplier=2):
            await asyncio.sleep(0.001)
            return x * multiplier

        sync_multiply = async_to_sync(async_multiply)
        result = sync_multiply(5, multiplier=3)
        assert result == 15

    def test_async_to_sync_exception_handling(self):
        """Test that exceptions in async functions are properly propagated."""

        async def async_error():
            await asyncio.sleep(0.001)
            raise ValueError("Test error")

        sync_error = async_to_sync(async_error)
        with pytest.raises(ValueError, match="Test error"):
            sync_error()

    def test_async_to_sync_return_types(self):
        """Test that return types are preserved."""

        async def async_return_dict():
            await asyncio.sleep(0.001)
            return {"key": "value", "number": 42}

        sync_return_dict = async_to_sync(async_return_dict)
        result = sync_return_dict()
        assert isinstance(result, dict)
        assert result == {"key": "value", "number": 42}


class TestGetTestDirectory:
    """Test cases for the get_test_directory function."""

    def test_get_test_directory_exists(self):
        """Test that get_test_directory creates a directory that exists."""
        test_dir = get_test_directory()
        assert os.path.exists(test_dir)
        assert os.path.isdir(test_dir)

    def test_get_test_directory_in_temp(self):
        """Test that test directory is created in system temp directory."""
        test_dir = get_test_directory()
        temp_root = tempfile.gettempdir()
        assert test_dir.startswith(temp_root)

    def test_get_test_directory_unique(self):
        """Test that get_test_directory creates unique directories."""
        dirs = set()
        for _ in range(5):
            test_dir = get_test_directory()
            assert test_dir not in dirs, "Generated duplicate directory path"
            dirs.add(test_dir)

    def test_get_test_directory_naming_pattern(self):
        """Test that test directory follows expected naming pattern."""
        test_dir = get_test_directory()
        dir_name = os.path.basename(test_dir)
        assert dir_name.startswith("ragas_test_")
        # The suffix should be the nano_id, which is alphanumeric
        suffix = dir_name[len("ragas_test_") :]
        assert suffix.isalnum()

    def test_get_test_directory_writable(self):
        """Test that the created test directory is writable."""
        test_dir = get_test_directory()
        # Try to create a file in the directory
        test_file = os.path.join(test_dir, "test_file.txt")
        with open(test_file, "w") as f:
            f.write("test content")

        # Verify file was created and has correct content
        assert os.path.exists(test_file)
        with open(test_file, "r") as f:
            content = f.read()
        assert content == "test content"

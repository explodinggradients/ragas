import pytest

from ragas.utils import batched, check_if_sum_is_close, get_from_dict


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

import pytest

from ragas.utils import check_if_sum_is_close, get_from_dict


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

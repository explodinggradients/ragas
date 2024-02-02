import pytest

from ragas.utils import check_if_sum_is_close


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

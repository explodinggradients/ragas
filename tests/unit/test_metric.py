import pytest

from ragas.metrics.base import make_batches


@pytest.mark.parametrize(
    "batch_size, total_size, len_expected", [(5, 10, 2), (5, 11, 3), (5, 9, 2)]
)
def test_make_batches(batch_size, total_size, len_expected):
    batches = make_batches(total_size, batch_size)
    assert len(batches) == len_expected

import numpy as np
import pytest

from ragas.run_config import RunConfig


def test_random_num_generator():
    rc = RunConfig(seed=32)
    assert isinstance(rc.rng, np.random.Generator)
    assert rc.rng.random() == pytest.approx(0.160, rel=1e2)

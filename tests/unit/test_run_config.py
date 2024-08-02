from ragas.run_config import RunConfig
import pytest
import numpy as np


def test_random_num_generator():
    rc = RunConfig(seed=32)
    assert isinstance(rc.rng, np.random.Generator)
    assert rc.rng.random() == pytest.approx(0.160, rel=1e2)

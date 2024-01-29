from __future__ import annotations

import re
import warnings

import numpy as np

rng = np.random.default_rng(seed=42)


def load_as_score(text):
    """
    validate and returns given text as score
    """

    pattern = r"^[\d.]+$"
    if not re.match(pattern, text):
        warnings.warn("Invalid score")
        score = 0.0
    else:
        score = eval(text)

    return score

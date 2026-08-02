from __future__ import annotations

import typing as t

import pytest

from ragas.metrics._context_precision import (
    LLMContextPrecisionWithReference,
    Verification,
)


def _v(verdict: int, reason: str = "r") -> Verification:
    return Verification(reason=reason, verdict=verdict)


def _ap(metric: LLMContextPrecisionWithReference, verdicts: t.List[int]) -> float:
    return metric._calculate_average_precision([_v(v) for v in verdicts])


@pytest.fixture
def metric() -> LLMContextPrecisionWithReference:
    # _calculate_average_precision is a pure function of the verdicts list and
    # does not touch self.llm, so a bare instance is enough.
    return LLMContextPrecisionWithReference()


def test_perfect_retrieval_returns_exactly_one(metric):
    """A perfectly ranked retrieval must score exactly 1.0, not 0.9999999999.

    The previous unconditional +1e-10 in the denominator made every non-zero
    score fall short of 1.0; this is the regression test for #2909.
    """
    assert _ap(metric, [1, 1, 1]) == 1.0
    assert _ap(metric, [1]) == 1.0
    assert _ap(metric, [1] * 20) == 1.0


def test_no_relevant_returns_zero(metric):
    """Zero relevant items: score is exactly 0.0 (no epsilon, no NaN)."""
    assert _ap(metric, [0, 0, 0]) == 0.0
    assert _ap(metric, [0]) == 0.0


def test_empty_verifications_returns_zero(metric):
    """Empty input: 0.0 (no items, no relevant, score 0.0)."""
    assert _ap(metric, []) == 0.0


@pytest.mark.parametrize(
    "verdicts, expected",
    [
        # [1,0,1] — relevant at ranks 1 and 3
        # AP = (1/1 + 2/3) / 2 = (1 + 0.6667) / 2 = 0.8333
        ([1, 0, 1], 5 / 6),
        # [0,1,1] — relevant at ranks 2 and 3
        # AP = (1/2 + 2/3) / 2 = (0.5 + 0.6667) / 2 = 0.5833
        ([0, 1, 1], 7 / 12),
        # [1,0,0,1,1,0,1] — textbook example, AP = 0.69...
        # AP = (1/1 + 2/4 + 3/5 + 4/7) / 4 = (1 + 0.5 + 0.6 + 0.5714) / 4
        ([1, 0, 0, 1, 1, 0, 1], (1 + 0.5 + 0.6 + 4 / 7) / 4),
    ],
)
def test_average_precision_matches_textbook_formula(metric, verdicts, expected):
    """Non-trivial rankings must match the standard AP formula exactly.

    The previous `+ 1e-10` epsilon silently scaled every non-zero score by
    cumsum / (cumsum + 1e-10), so cross-checks against any external
    average-precision implementation (e.g., sklearn) would always fail.
    """
    assert _ap(metric, verdicts) == pytest.approx(expected)


def test_average_precision_is_not_nan_on_empty(metric):
    """Empty input returns 0.0 (not NaN) — the np.isnan warning below the
    fix was unreachable dead code that the maintainer should remove or
    relocate; this test guards against any future regression that
    re-introduces a NaN-producing path.
    """
    score = _ap(metric, [])
    assert score == 0.0
    import math

    assert not math.isnan(score)

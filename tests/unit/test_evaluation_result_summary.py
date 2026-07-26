"""Tests for EvaluationResult.summary() (the new per-metric aggregate helper).

Covers the new ``summary()`` method on :class:`EvaluationResult`. The
tests bypass :meth:`EvaluationResult.__post_init__` (which calls
``parse_run_traces`` and would require a populated ``ragas_traces``
dict) by constructing the dataclass via ``__new__`` and seeding
``_scores_dict`` directly.
"""

from __future__ import annotations

import math
import typing as t

from ragas.dataset_schema import EvaluationResult
from ragas.utils import safe_nanmean


def _make_result(scores: t.List[t.Dict[str, t.Any]]) -> EvaluationResult:
    """Build an EvaluationResult with the given per-row scores, bypassing
    ``__post_init__`` so the tests do not need a populated ``ragas_traces``
    dict. The method under test (``summary``) only depends on
    ``_scores_dict``, which we seed directly from the input ``scores``.
    """
    result = EvaluationResult.__new__(EvaluationResult)
    result.scores = scores
    result._scores_dict = {k: [d[k] for d in scores] for k in scores[0].keys()}
    # Touch safe_nanmean so a future change to its signature is caught
    # at import/test time, not by an unrelated unit.
    _ = safe_nanmean([])
    return result


def test_summary_returns_mean_count_missing_per_metric() -> None:
    result = _make_result(
        scores=[
            {"faithfulness": 0.9, "answer_relevancy": 0.8},
            {"faithfulness": 0.7, "answer_relevancy": 0.6},
        ]
    )

    summary = result.summary()
    assert set(summary.keys()) == {"faithfulness", "answer_relevancy"}
    assert summary["faithfulness"] == {"mean": 0.8, "count": 2, "missing": 0}
    assert summary["answer_relevancy"] == {
        "mean": 0.7,
        "count": 2,
        "missing": 0,
    }


def test_summary_count_and_missing_sum_to_row_count() -> None:
    scores = [
        {"faithfulness": 0.9, "answer_relevancy": 0.8},
        {"faithfulness": 0.7, "answer_relevancy": None},
        {"faithfulness": None, "answer_relevancy": 0.5},
        {"faithfulness": 0.6, "answer_relevancy": 0.4},
    ]
    result = _make_result(scores=scores)

    summary = result.summary()
    for metric in ("faithfulness", "answer_relevancy"):
        assert summary[metric]["count"] + summary[metric]["missing"] == len(scores)


def test_summary_nan_in_column_does_not_poison_aggregate() -> None:
    # safe_nanmean drops NaN; the surviving values still average.
    scores = [
        {"faithfulness": 0.9},
        {"faithfulness": float("nan")},
        {"faithfulness": 0.7},
    ]
    result = _make_result(scores=scores)

    summary = result.summary()
    assert summary["faithfulness"]["count"] == 2
    assert summary["faithfulness"]["missing"] == 1
    assert math.isclose(summary["faithfulness"]["mean"], 0.8, rel_tol=1e-9)


def test_summary_all_null_column_returns_nan_mean() -> None:
    scores = [
        {"faithfulness": None},
        {"faithfulness": None},
    ]
    result = _make_result(scores=scores)

    summary = result.summary()
    assert summary["faithfulness"]["count"] == 0
    assert summary["faithfulness"]["missing"] == 2
    assert math.isnan(summary["faithfulness"]["mean"])


def test_summary_binary_column_aggregates_fraction_of_ones() -> None:
    # AspectCritic-style verdicts are 0 or 1; mean == fraction of 1s.
    scores = [
        {"aspect": 1},
        {"aspect": 1},
        {"aspect": 0},
        {"aspect": 1},
    ]
    result = _make_result(scores=scores)

    summary = result.summary()
    assert math.isclose(summary["aspect"]["mean"], 0.75, rel_tol=1e-9)
    assert summary["aspect"] == {"mean": 0.75, "count": 4, "missing": 0}


def test_summary_is_idempotent() -> None:
    # Calling summary() twice returns equivalent dicts (no shared state).
    result = _make_result(scores=[{"faithfulness": 0.5}, {"faithfulness": 0.5}])
    first = result.summary()
    second = result.summary()
    assert first == second

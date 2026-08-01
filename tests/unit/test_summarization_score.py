"""Tests for the legacy SummarizationScore metric in
src/ragas/metrics/_summarization.py.

Covers the empty-`answers` path in `_compute_qa_score` that previously
raised ZeroDivisionError and aborted the entire evaluation. The fix
adds an early-return guard so the empty path returns 0.0 instead of
crashing.
"""

import pytest

from ragas.metrics._summarization import SummarizationScore


class TestSummarizationScoreQAScore:
    """Tests for SummarizationScore._compute_qa_score.

    The QA score is the fraction of answers equal to "1" (case
    insensitive). It divides by `len(answers)`, which raises when
    `answers` is empty. The early-return guard returns 0.0 for the
    empty case.
    """

    def test_empty_answers_returns_zero(self):
        """Regression: empty `answers` used to raise ZeroDivisionError.
        Now it returns 0.0 (consistent with the `+ 1e-10` pattern in
        `_compute_conciseness_score`).
        """
        metric = SummarizationScore()
        # This used to raise ZeroDivisionError. After the fix, it
        # returns 0.0 — a clear "no signal" value that the metric's
        # _compute_score can fold into a final 0.0 (or near-zero with
        # conciseness contribution).
        assert metric._compute_qa_score([]) == 0.0

    def test_all_zero_answers_returns_zero(self):
        """Sanity: if every answer is "0", the QA score is 0.0
        (no correct answers), not a divide-by-zero."""
        metric = SummarizationScore()
        assert metric._compute_qa_score(["0", "0", "0"]) == 0.0

    def test_all_one_answers_returns_one(self):
        """Sanity: if every answer is "1", the QA score is 1.0."""
        metric = SummarizationScore()
        assert metric._compute_qa_score(["1", "1", "1"]) == 1.0

    def test_mixed_answers_returns_correct_fraction(self):
        """Sanity: the fraction is `correct / total`."""
        metric = SummarizationScore()
        # 2 correct out of 3 → 2/3
        assert metric._compute_qa_score(["1", "1", "0"]) == pytest.approx(2 / 3)

    def test_answers_case_insensitive(self):
        """Sanity: 'a.lower() == "1"' means the comparison is
        case-insensitive on the LLM's side; verify it.
        """
        metric = SummarizationScore()
        assert metric._compute_qa_score(["1", "Yes", "no"]) == pytest.approx(1 / 3)
        # 'Yes' and 'no' are not '1' → only the literal '1' counts.

    def test_empty_answers_does_not_raise(self):
        """Regression assertion (negative form): the empty path MUST
        NOT raise. This is the contract the maintainers preserved as
        'legacy behavior' in the modern collections version.
        """
        metric = SummarizationScore()
        try:
            result = metric._compute_qa_score([])
        except ZeroDivisionError as e:
            pytest.fail(
                f"_compute_qa_score([]) raised ZeroDivisionError: {e}. "
                f"Empty answers should return 0.0, not crash."
            )
        assert result == 0.0

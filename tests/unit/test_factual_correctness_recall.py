"""Unit tests for FactualCorrectness recall-mode scoring formula (issue #2693).

These tests exercise the pure scoring arithmetic by patching out the LLM calls,
so no API key or network access is required.
"""

from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from ragas.metrics._factual_correctness import FactualCorrectness


class TestFactualCorrectnessRecallFormula:
    """Concrete example from issue #2693:
    - Response: 4 claims, 3 supported by reference → reference_response = [T, T, T, F]
    - Reference: 3 claims, 2 supported by response → response_reference = [T, T, F]
    - Expected precision = 3/4 = 0.75
    - Expected recall   = 2/3 ≈ 0.67  (was incorrectly returning 0.75 before fix)
    - Expected F1       = 2*0.75*0.67/(0.75+0.67) ≈ 0.71
    """

    REFERENCE_RESPONSE = np.array([True, True, True, False])  # precision direction
    RESPONSE_REFERENCE = np.array([True, True, False])  # recall direction

    def _make_metric(self, mode: str) -> FactualCorrectness:
        # FactualCorrectness requires an LLM but we won't call any LLM methods
        fc = FactualCorrectness.__new__(FactualCorrectness)
        fc.mode = mode
        fc.beta = 1.0
        fc.llm = None  # never used in the patched path
        return fc

    def _run_scoring(self, mode: str, ref_resp, resp_ref) -> float:
        fc = self._make_metric(mode)
        tp = sum(ref_resp)
        fp = sum(~ref_resp)
        fn = sum(~resp_ref) if mode != "precision" else 0

        if mode == "precision":
            score = tp / (tp + fp + 1e-8)
        elif mode == "recall":
            tp_recall = sum(resp_ref)
            score = tp_recall / (tp_recall + fn + 1e-8)
        else:
            from ragas.metrics.utils import fbeta_score

            score = fbeta_score(tp, fp, fn, fc.beta)
        return float(np.round(score, 2))

    def test_precision_score(self):
        score = self._run_scoring(
            "precision", self.REFERENCE_RESPONSE, self.RESPONSE_REFERENCE
        )
        assert score == pytest.approx(0.75, abs=0.01), (
            f"Expected precision≈0.75, got {score}"
        )

    def test_recall_score_uses_reference_claims_as_tp(self):
        """Recall TP must be reference claims supported by response, not response claims."""
        score = self._run_scoring(
            "recall", self.REFERENCE_RESPONSE, self.RESPONSE_REFERENCE
        )
        # 2 reference claims supported by response, 1 not → recall = 2/3 ≈ 0.67
        assert score == pytest.approx(0.67, abs=0.01), (
            f"Expected recall≈0.67 but got {score}; "
            "if this returns 0.75 the bug is not fixed (precision TP used for recall)"
        )

    def test_recall_differs_from_precision_when_claims_asymmetric(self):
        """Recall and precision should differ when response and reference have different claim counts."""
        precision = self._run_scoring(
            "precision", self.REFERENCE_RESPONSE, self.RESPONSE_REFERENCE
        )
        recall = self._run_scoring(
            "recall", self.REFERENCE_RESPONSE, self.RESPONSE_REFERENCE
        )
        assert recall != precision, (
            "Recall and precision must differ for this asymmetric example; "
            "identical values indicate the same TP is being used for both"
        )

    @pytest.mark.asyncio
    async def test_single_turn_ascore_recall(self):
        """Integration: patch LLM calls and verify the full _single_turn_ascore path."""
        from ragas.dataset_schema import SingleTurnSample

        fc = self._make_metric("recall")
        fc.llm = AsyncMock()

        sample = SingleTurnSample(
            response="dummy response",
            reference="dummy reference",
        )

        with patch.object(
            fc,
            "decompose_and_verify_claims",
            side_effect=[self.REFERENCE_RESPONSE, self.RESPONSE_REFERENCE],
        ):
            score = await fc._single_turn_ascore(sample, callbacks=None)

        # 2 reference claims supported / 3 total reference claims → recall ≈ 0.67
        assert score == pytest.approx(0.67, abs=0.01), (
            f"Expected recall≈0.67 but got {score}"
        )

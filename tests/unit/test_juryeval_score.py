"""Tests for JuryEvalScore metric."""

import importlib
import sys

import pytest

# Shim for a pre-existing broken import in ragas.llms.base
_vertexai = type(sys)('langchain_community.chat_models.vertexai')
_vertexai.ChatVertexAI = object
sys.modules['langchain_community.chat_models.vertexai'] = _vertexai

from ragas.metrics.collections._juryeval_score import JuryEvalScore


class TestJuryEvalScore:
    def test_init_default_values(self):
        metric = JuryEvalScore()
        assert metric.name == "juryeval_score"
        assert metric.judge_type == "pointwise"
        assert metric.judge_model == "gpt-4"

    def test_init_custom_values(self):
        metric = JuryEvalScore(
            judge_type="pairwise",
            judge_model="gpt-4o",
            temperature=0.5,
            max_retries=5,
            name="custom_juryeval",
        )
        assert metric.judge_type == "pairwise"
        assert metric.judge_model == "gpt-4o"
        assert metric.temperature == 0.5
        assert metric.max_retries == 5
        assert metric.name == "custom_juryeval"

    @pytest.mark.asyncio
    async def test_pointwise_scoring(self, mocker):
        mock_judge = mocker.patch("juryeval.PointwiseJudge")
        instance = mock_judge.return_value
        instance.score.return_value = {"score": 0.85, "reason": "Good answer"}

        metric = JuryEvalScore(judge_type="pointwise", max_retries=1)
        result = await metric.ascore(user_input="test q", response="test a")

        assert result.value == 0.85
        assert result.reason == "Good answer"
        instance.score.assert_called_once_with(
            output="test a", question="test q", reference=None
        )

    @pytest.mark.asyncio
    async def test_pairwise_scoring(self, mocker):
        mock_judge = mocker.patch("juryeval.PairwiseJudge")
        instance = mock_judge.return_value
        instance.compare.return_value = {"score": 1.0, "reason": "A is better"}

        metric = JuryEvalScore(judge_type="pairwise", max_retries=1)
        result = await metric.ascore(
            user_input="test q", response="ans a", reference="ans b"
        )

        assert result.value == 1.0
        assert result.reason == "A is better"
        instance.compare.assert_called_once_with(
            answer_a="ans a", answer_b="ans b", question="test q"
        )

    @pytest.mark.asyncio
    async def test_unknown_judge_type(self):
        metric = JuryEvalScore(judge_type="unknown", max_retries=1)
        with pytest.raises(ValueError, match="Unknown judge_type"):
            await metric.ascore(user_input="q", response="a")

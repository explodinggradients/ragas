"""Unit tests for RougeScore v2."""

import pytest

from ragas.metrics.v2 import RougeScore


class TestRougeScore:
    """Test suite for RougeScore v2 metric."""

    def test_initialization_defaults(self):
        """Test initialization with default parameters."""
        metric = RougeScore()

        assert metric.name == "rouge_score_v2"
        assert metric.rouge_type == "rougeL"
        assert metric.mode == "fmeasure"
        assert metric.allowed_values == (0.0, 1.0)

    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        metric = RougeScore(
            rouge_type="rouge1",
            mode="precision",
        )

        assert metric.rouge_type == "rouge1"
        assert metric.mode == "precision"

    @pytest.mark.asyncio
    async def test_ascore_identical_texts(self):
        """Test scoring with identical reference and response."""
        metric = RougeScore()
        result = await metric.ascore(
            reference="The quick brown fox jumps over the lazy dog",
            response="The quick brown fox jumps over the lazy dog",
        )

        # Identical texts should have perfect score
        assert result.value == 1.0

    @pytest.mark.asyncio
    async def test_ascore_completely_different_texts(self):
        """Test scoring with completely different texts."""
        metric = RougeScore()
        result = await metric.ascore(
            reference="The quick brown fox",
            response="Completely different words here",
        )

        # Completely different texts should have low score
        assert result.value < 0.3

    @pytest.mark.asyncio
    async def test_ascore_partial_overlap(self):
        """Test scoring with partial overlap."""
        metric = RougeScore()
        result = await metric.ascore(
            reference="The capital of France is Paris",
            response="Paris is the capital of France",
        )

        # Partial overlap should have medium score
        assert 0.3 < result.value < 1.0

    @pytest.mark.asyncio
    async def test_ascore_rouge1_vs_rougeL(self):
        """Test difference between rouge1 and rougeL."""
        reference = "The quick brown fox"
        response = "brown fox quick The"  # Same words, different order

        rouge1_metric = RougeScore(rouge_type="rouge1")
        rougeL_metric = RougeScore(rouge_type="rougeL")

        rouge1_result = await rouge1_metric.ascore(
            reference=reference, response=response
        )
        rougeL_result = await rougeL_metric.ascore(
            reference=reference, response=response
        )

        # rouge1 should score higher (unigram match)
        # rougeL should score lower (LCS cares about order)
        assert rouge1_result.value == 1.0  # All unigrams match
        assert rougeL_result.value < rouge1_result.value

    @pytest.mark.asyncio
    async def test_ascore_different_modes(self):
        """Test different scoring modes."""
        reference = "The quick brown fox jumps"
        response = "The quick brown"  # Subset of reference

        fmeasure_metric = RougeScore(mode="fmeasure")
        precision_metric = RougeScore(mode="precision")
        recall_metric = RougeScore(mode="recall")

        fmeasure_result = await fmeasure_metric.ascore(
            reference=reference, response=response
        )
        precision_result = await precision_metric.ascore(
            reference=reference, response=response
        )
        recall_result = await recall_metric.ascore(
            reference=reference, response=response
        )

        # Precision should be high (all response words in reference)
        assert precision_result.value > 0.9

        # Recall should be lower (not all reference words in response)
        assert recall_result.value < precision_result.value

        # F-measure should be between precision and recall
        assert recall_result.value <= fmeasure_result.value <= precision_result.value

    @pytest.mark.asyncio
    async def test_abatch_score(self):
        """Test batch scoring."""
        metric = RougeScore()
        inputs = [
            {"reference": "Text A", "response": "Text A"},
            {"reference": "Text B", "response": "Different"},
            {"reference": "Text C", "response": "Text C"},
        ]

        results = await metric.abatch_score(inputs)

        assert len(results) == 3
        assert results[0].value == 1.0  # Perfect match
        assert results[1].value < 0.5  # Low match
        assert results[2].value == 1.0  # Perfect match

    def test_score_sync(self):
        """Test synchronous scoring."""
        metric = RougeScore()
        result = metric.score(
            reference="Hello world",
            response="Hello world",
        )

        assert result.value == 1.0

    def test_serialization(self):
        """Test Pydantic serialization."""
        metric = RougeScore(rouge_type="rouge1", mode="precision")

        # Test model_dump
        data = metric.model_dump()
        assert data["name"] == "rouge_score_v2"
        assert data["rouge_type"] == "rouge1"
        assert data["mode"] == "precision"

        # Test model_dump_json
        import json

        json_str = metric.model_dump_json()
        loaded_data = json.loads(json_str)
        assert loaded_data["rouge_type"] == "rouge1"

        # Test reconstruction
        new_metric = RougeScore(**data)
        assert new_metric.rouge_type == metric.rouge_type
        assert new_metric.mode == metric.mode

    def test_missing_dependency_error(self):
        """Test error when rouge_score package is missing."""
        import sys
        from unittest.mock import patch

        metric = RougeScore()

        # Mock missing rouge_score package
        with patch.dict(sys.modules, {"rouge_score": None}):
            with pytest.raises(ImportError, match="rouge_score is required"):
                metric.score(reference="test", response="test")

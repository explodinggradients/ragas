import pytest

from ragas.metrics.collections._bleu_score import BleuScore
from ragas.metrics.utils import split_bleu_sentences


class TestSplitBleuSentences:
    def test_english_period_space_unchanged(self):
        text = "The cat sat on the mat. The dog ran in the park."
        assert split_bleu_sentences(text) == text.split(". ")

    def test_multiline_splits_by_line(self):
        text = "First line\nSecond line"
        assert split_bleu_sentences(text) == ["First line", "Second line"]

    def test_single_sentence_without_period_space(self):
        text = "Die Hauptstadt ist Berlin"
        assert split_bleu_sentences(text) == [text]

    def test_empty_string(self):
        assert split_bleu_sentences("") == [""]


@pytest.mark.asyncio
async def test_bleu_score_multiline_reference_and_response():
    metric = BleuScore()
    reference = "Paris is the capital.\nIt is in France."
    response = "Paris is the capital.\nIt is in France."
    result = await metric.ascore(reference=reference, response=response)
    assert result.value == 1.0


@pytest.mark.asyncio
async def test_bleu_score_english_period_space_backward_compatible():
    metric = BleuScore()
    reference = "The cat sat on the mat. The dog ran in the park."
    response = "The cat sat on the mat. The dog ran in the park."
    result = await metric.ascore(reference=reference, response=response)
    assert result.value == 1.0

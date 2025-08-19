import gzip
import json
import warnings

import pytest
from pydantic import BaseModel

from ragas.experimental.prompt.base import Prompt


class MockResponseModel(BaseModel):
    """Mock Pydantic model for testing response_model functionality."""

    answer: str
    confidence: float = 0.9

    model_config = {
        "json_schema_extra": {"example": {"answer": "Test answer", "confidence": 0.95}}
    }


class TestPromptSaveLoad:
    """Test suite for Prompt save/load functionality."""

    def test_save_load_basic_without_response_model(self, tmp_path):
        """Test basic save/load functionality without response_model."""
        # Create a prompt with examples
        original = Prompt(
            instruction="Answer the question: {question}",
            examples=[
                ({"question": "What is 2+2?"}, {"answer": "4"}),
                ({"question": "What is the capital of France?"}, {"answer": "Paris"}),
            ],
        )

        # Test save to regular JSON
        json_path = tmp_path / "test_prompt.json"
        original.save(str(json_path))

        # Verify file was created and contains expected data
        assert json_path.exists()
        with open(json_path, "r") as f:
            data = json.load(f)

        assert data["type"] == "Prompt"
        assert data["format_version"] == "1.0"
        assert data["instruction"] == "Answer the question: {question}"
        assert len(data["examples"]) == 2
        assert data["response_model_info"] is None

        # Test load
        loaded = Prompt.load(str(json_path))

        assert loaded.instruction == original.instruction
        assert loaded.examples == original.examples
        assert loaded.response_model is None

    def test_save_load_with_gzip_compression(self, tmp_path):
        """Test save/load with gzip compression."""
        original = Prompt(
            instruction="Compressed prompt: {input}",
            examples=[({"input": "test"}, {"output": "result"})],
        )

        # Test save with .gz extension
        gz_path = tmp_path / "compressed_prompt.json.gz"
        original.save(str(gz_path))

        # Verify compressed file exists and can be read
        assert gz_path.exists()
        with gzip.open(gz_path, "rt", encoding="utf-8") as f:
            data = json.load(f)
        assert data["type"] == "Prompt"

        # Test load from compressed file
        loaded = Prompt.load(str(gz_path))
        assert loaded.instruction == original.instruction
        assert loaded.examples == original.examples

    def test_save_with_response_model_shows_warning(self, tmp_path):
        """Test that saving with response_model shows appropriate warning."""
        mock_model = MockResponseModel(answer="test")
        prompt = Prompt(instruction="Test: {input}", response_model=mock_model)

        json_path = tmp_path / "prompt_with_model.json"

        # Capture warnings during save
        with pytest.warns(UserWarning, match="response_model cannot be saved"):
            prompt.save(str(json_path))

        # Verify response_model_info was saved
        with open(json_path, "r") as f:
            data = json.load(f)

        assert data["response_model_info"] is not None
        assert data["response_model_info"]["class_name"] == "MockResponseModel"
        assert "schema" in data["response_model_info"]
        assert (
            data["response_model_info"]["note"]
            == "You must provide this model when loading"
        )

    def test_load_requires_response_model_when_expected(self, tmp_path):
        """Test error when response_model is required but not provided."""
        # Create and save a prompt with response_model
        mock_model = MockResponseModel(answer="test")
        prompt = Prompt("Test: {input}", response_model=mock_model)

        json_path = tmp_path / "model_required.json"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore the save warning for this test
            prompt.save(str(json_path))

        # Try to load without providing response_model - should raise error
        with pytest.raises(ValueError, match="requires a response_model"):
            Prompt.load(str(json_path))

        # Verify error message contains helpful information
        with pytest.raises(ValueError, match="MockResponseModel"):
            Prompt.load(str(json_path))

    def test_load_with_response_model_succeeds(self, tmp_path):
        """Test successful load when response_model is provided."""
        # Create and save a prompt with response_model
        mock_model = MockResponseModel(answer="test")
        original = Prompt("Test: {input}", response_model=mock_model)

        json_path = tmp_path / "with_model.json"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            original.save(str(json_path))

        # Load with response_model provided
        new_model = MockResponseModel(answer="different")
        loaded = Prompt.load(str(json_path), response_model=new_model)

        assert loaded.instruction == original.instruction
        assert loaded.response_model == new_model

    def test_response_model_schema_validation_warning(self, tmp_path):
        """Test warning when provided response_model schema differs from saved."""

        # Create a different model with different schema
        class DifferentModel(BaseModel):
            result: str  # Different field name
            score: int  # Different field type

        # Save with MockResponseModel
        mock_model = MockResponseModel(answer="test")
        prompt = Prompt("Test: {input}", response_model=mock_model)

        json_path = tmp_path / "schema_test.json"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prompt.save(str(json_path))

        # Load with different model - should show warning
        different_model = DifferentModel(result="test", score=1)
        with pytest.warns(UserWarning, match="schema differs"):
            Prompt.load(str(json_path), response_model=different_model)

    def test_file_validation_errors(self, tmp_path):
        """Test various file validation error conditions."""
        # Test loading non-existent file
        with pytest.raises(ValueError, match="Cannot load prompt"):
            Prompt.load("nonexistent.json")

        # Test loading invalid JSON
        invalid_json_path = tmp_path / "invalid.json"
        with open(invalid_json_path, "w") as f:
            f.write("invalid json content")

        with pytest.raises(ValueError, match="Cannot load prompt"):
            Prompt.load(str(invalid_json_path))

        # Test loading wrong file type
        wrong_type_path = tmp_path / "wrong_type.json"
        with open(wrong_type_path, "w") as f:
            json.dump({"type": "NotAPrompt", "instruction": "test"}, f)

        with pytest.raises(ValueError, match="File is not a Prompt"):
            Prompt.load(str(wrong_type_path))

    def test_save_file_permission_error(self, tmp_path):
        """Test error handling when save location is not writable."""
        prompt = Prompt("Test: {input}")

        # Try to save to non-existent directory (should raise error)
        invalid_path = tmp_path / "nonexistent_dir" / "test.json"
        with pytest.raises(ValueError, match="Cannot save prompt"):
            prompt.save(str(invalid_path))

    def test_round_trip_preserves_data(self, tmp_path):
        """Test that save/load round-trip preserves all data correctly."""
        original = Prompt(
            instruction="Complex instruction with {param1} and {param2}",
            examples=[
                ({"param1": "value1", "param2": "value2"}, {"result": "output1"}),
                (
                    {"param1": "test", "param2": "data"},
                    {"result": "output2", "extra": "info"},
                ),
            ],
        )

        # Save and load
        json_path = tmp_path / "round_trip.json"
        original.save(str(json_path))
        loaded = Prompt.load(str(json_path))

        # Verify all data is preserved
        assert loaded.instruction == original.instruction
        assert loaded.examples == original.examples
        assert loaded.response_model == original.response_model

        # Verify formatting works the same
        test_params = {"param1": "test1", "param2": "test2"}
        assert loaded.format(**test_params) == original.format(**test_params)

    def test_empty_examples_handling(self, tmp_path):
        """Test handling of prompts with no examples."""
        prompt = Prompt("Simple instruction: {input}")

        json_path = tmp_path / "no_examples.json"
        prompt.save(str(json_path))
        loaded = Prompt.load(str(json_path))

        assert loaded.instruction == prompt.instruction
        assert loaded.examples == []
        assert loaded.format(input="test") == "Simple instruction: test"

    def test_unicode_characters_handling(self, tmp_path):
        """Test that save/load correctly handles unicode characters, emojis, and international text."""
        # Create prompt with unicode instruction and examples
        unicode_prompt = Prompt(
            instruction="Répondez à la question en {language}: {question} 🤔",
            examples=[
                # Mixed languages with emojis
                (
                    {"question": "¿Qué es 数学?", "language": "français"},
                    {"answer": "Les mathématiques! 📊", "confidence": "très élevée"},
                ),
                # Korean characters
                (
                    {"question": "안녕하세요?", "language": "English"},
                    {"answer": "Hello in Korean! 🇰🇷", "greeting": "안녕"},
                ),
                # Arabic and mathematical symbols
                (
                    {"question": "ما هو π؟", "language": "العربية"},
                    {"answer": "π ≈ 3.14159... ∞", "symbol": "π"},
                ),
                # Emojis and special characters
                (
                    {"question": "What's the weather? ☀️🌧️", "language": "emoji"},
                    {"answer": "Sunny with chance of rain! 🌤️⛈️", "mood": "🌈"},
                ),
            ],
        )

        # Test with regular JSON
        json_path = tmp_path / "unicode_prompt.json"
        unicode_prompt.save(str(json_path))

        # Verify file contains unicode (JSON escapes unicode as \u sequences)
        with open(json_path, "r", encoding="utf-8") as f:
            file_content = f.read()
            # Check that unicode characters are properly represented in JSON
            # JSON uses \u escape sequences for non-ASCII characters
            assert "\\u00e9" in file_content  # é in Répondez
            assert "\\u6570\\u5b66" in file_content  # 数学
            assert "\\ud83e\\udd14" in file_content  # 🤔 emoji
            assert "\\uc548\\ub155" in file_content  # 안녕

        # Load and verify all unicode is preserved
        loaded = Prompt.load(str(json_path))

        assert loaded.instruction == unicode_prompt.instruction
        assert loaded.examples == unicode_prompt.examples

        # Test formatting with unicode parameters
        formatted = loaded.format(
            question="Comment allez-vous? 😊", language="français"
        )
        # Should contain the formatted instruction
        expected_instruction = (
            "Répondez à la question en français: Comment allez-vous? 😊 🤔"
        )
        assert expected_instruction in formatted
        # Should also contain examples since the prompt has examples
        assert "Examples:" in formatted

        # Test with gzip compression
        gz_path = tmp_path / "unicode_prompt.json.gz"
        unicode_prompt.save(str(gz_path))

        # Load from compressed file
        loaded_gz = Prompt.load(str(gz_path))

        assert loaded_gz.instruction == unicode_prompt.instruction
        assert loaded_gz.examples == unicode_prompt.examples

        # Verify both loaded versions are identical
        assert loaded.instruction == loaded_gz.instruction
        assert loaded.examples == loaded_gz.examples

        # Test round-trip with various unicode scenarios
        test_cases = [
            {"question": "Здравствуйте! 🇷🇺", "language": "русский"},  # Russian
            {"question": "こんにちは 🇯🇵", "language": "日本語"},  # Japanese
            {"question": "∑∫∂∆∇∞ ≠ ≤ ≥", "language": "math"},  # Mathematical symbols
            {"question": "🚀🌟💡🎯🔥", "language": "emoji"},  # Pure emojis
        ]

        for test_case in test_cases:
            formatted_result = loaded.format(**test_case)
            # Verify formatting works and contains the unicode input
            assert test_case["question"] in formatted_result
            assert test_case["language"] in formatted_result
            assert "🤔" in formatted_result  # Original emoji from instruction

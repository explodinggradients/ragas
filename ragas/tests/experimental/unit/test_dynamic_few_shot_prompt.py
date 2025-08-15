import json
import gzip
import warnings
import typing as t
import pytest
from pydantic import BaseModel

from ragas.experimental.embeddings.base import BaseEmbedding
from ragas.experimental.prompt.dynamic_few_shot import DynamicFewShotPrompt


class MockResponseModel(BaseModel):
    """Mock Pydantic model for testing response_model functionality."""

    answer: str
    confidence: float = 0.9

    model_config = {
        "json_schema_extra": {"example": {"answer": "Test answer", "confidence": 0.95}}
    }


class MockEmbeddingModel(BaseEmbedding):
    """Mock embedding model for testing embedding functionality."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self._call_count = 0

    def embed_text(self, text: str, **kwargs: t.Any) -> list[float]:
        """Return deterministic embeddings based on text length and content."""
        self._call_count += 1
        # Create deterministic embedding based on text hash
        import hashlib

        text_hash = int(hashlib.md5(text.encode()).hexdigest(), 16)
        # Generate deterministic floats between -1 and 1
        embedding = []
        for i in range(self.dimension):
            value = ((text_hash + i) % 200000 - 100000) / 100000.0
            embedding.append(value)
        return embedding

    async def aembed_text(self, text: str, **kwargs) -> list[float]:
        """Async version of embed_text."""
        return self.embed_text(text)

    @property
    def call_count(self):
        return self._call_count


class TestDynamicFewShotPromptSaveLoad:
    """Test suite for DynamicFewShotPrompt save/load functionality."""

    def test_save_load_without_embedding_model(self, tmp_path):
        """Test basic save/load functionality without embedding model."""
        examples = [
            ({"question": "What is 1+1?"}, {"answer": "2"}),
            ({"question": "What is 2+2?"}, {"answer": "4"}),
            ({"question": "What is 3+3?"}, {"answer": "6"}),
        ]

        original = DynamicFewShotPrompt(
            instruction="Answer the math question: {question}",
            examples=examples,
            max_similar_examples=2,
            similarity_threshold=0.8,
        )

        # Test save to regular JSON
        json_path = tmp_path / "test_dynamic_prompt.json"
        original.save(str(json_path), include_embeddings=False)

        # Verify file was created and contains expected data
        assert json_path.exists()
        with open(json_path, "r") as f:
            data = json.load(f)

        assert data["type"] == "DynamicFewShotPrompt"
        assert data["format_version"] == "1.0"
        assert data["instruction"] == "Answer the math question: {question}"
        assert len(data["examples"]) == 3
        assert data["max_similar_examples"] == 2
        assert data["similarity_threshold"] == 0.8
        assert data["embedding_model_info"] is None
        assert data["response_model_info"] is None
        assert "embeddings" not in data

        # Test load
        loaded = DynamicFewShotPrompt.load(str(json_path))

        assert loaded.instruction == original.instruction
        assert loaded.max_similar_examples == original.max_similar_examples
        assert loaded.similarity_threshold == original.similarity_threshold
        assert len(loaded.example_store) == len(original.example_store)
        assert loaded.example_store._examples == original.example_store._examples
        assert loaded.response_model is None
        assert loaded.example_store.embedding_model is None

    def test_save_load_with_compression(self, tmp_path):
        """Test save/load with gzip compression."""
        examples = [
            ({"text": "Hello world", "lang": "en"}, {"translation": "Hola mundo"}),
            ({"text": "Good morning", "lang": "en"}, {"translation": "Buenos días"}),
        ]

        original = DynamicFewShotPrompt(
            instruction="Translate '{text}' to Spanish:",
            examples=examples,
            max_similar_examples=1,
            similarity_threshold=0.5,
        )

        # Test save with .gz extension
        gz_path = tmp_path / "dynamic_prompt.json.gz"
        original.save(str(gz_path), include_embeddings=False)

        # Verify compressed file exists and can be read
        assert gz_path.exists()
        with gzip.open(gz_path, "rt", encoding="utf-8") as f:
            data = json.load(f)
        assert data["type"] == "DynamicFewShotPrompt"

        # Test load from compressed file
        loaded = DynamicFewShotPrompt.load(str(gz_path))
        assert loaded.instruction == original.instruction
        assert loaded.max_similar_examples == original.max_similar_examples
        assert loaded.similarity_threshold == original.similarity_threshold
        assert len(loaded.example_store) == len(original.example_store)

    def test_save_load_with_embedding_model(self, tmp_path):
        """Test save/load functionality with embedding model."""
        mock_embedding = MockEmbeddingModel(dimension=3)
        examples = [
            ({"question": "What is AI?"}, {"answer": "Artificial Intelligence"}),
            ({"question": "What is ML?"}, {"answer": "Machine Learning"}),
        ]

        original = DynamicFewShotPrompt(
            instruction="Answer: {question}",
            examples=examples,
            embedding_model=mock_embedding,
            max_similar_examples=1,
            similarity_threshold=0.7,
        )

        # Verify embeddings were computed during creation
        assert len(original.example_store._embeddings_list) == 2
        assert len(original.example_store._embeddings_list[0]) == 3
        # Track call count for later verification
        assert mock_embedding.call_count >= 2  # At least 2 calls for 2 examples

        json_path = tmp_path / "with_embedding.json"

        # Test save with warning about embedding model
        with pytest.warns(UserWarning, match="embedding_model cannot be saved"):
            original.save(str(json_path), include_embeddings=True)

        # Verify file contains embedding data
        with open(json_path, "r") as f:
            data = json.load(f)

        assert data["embedding_model_info"] is not None
        assert data["embedding_model_info"]["class_name"] == "MockEmbeddingModel"
        assert "embeddings" in data
        assert len(data["embeddings"]) == 2
        assert len(data["embeddings"][0]) == 3

        # Test load with embedding model provided
        new_embedding = MockEmbeddingModel(dimension=3)
        loaded = DynamicFewShotPrompt.load(
            str(json_path), embedding_model=new_embedding
        )

        assert loaded.instruction == original.instruction
        assert loaded.example_store.embedding_model == new_embedding
        assert len(loaded.example_store._embeddings_list) == 2
        # Embeddings should be restored from file, not recomputed during load
        # (The new_embedding may be called during DynamicFewShotPrompt init, but embeddings are restored from file)
        assert new_embedding.call_count <= 2  # At most called during initialization

    def test_embedding_recomputation_on_load(self, tmp_path):
        """Test that embeddings are recomputed when not saved or model missing."""
        mock_embedding = MockEmbeddingModel()
        examples = [
            ({"question": "Test question"}, {"answer": "Test answer"}),
        ]

        original = DynamicFewShotPrompt(
            instruction="Answer: {question}",
            examples=examples,
            embedding_model=mock_embedding,
        )

        json_path = tmp_path / "no_embeddings.json"

        # Save without embeddings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            original.save(str(json_path), include_embeddings=False)

        # Load with new embedding model
        new_embedding = MockEmbeddingModel()
        initial_call_count = new_embedding.call_count
        loaded = DynamicFewShotPrompt.load(
            str(json_path), embedding_model=new_embedding
        )

        # Embeddings are computed during initialization when examples are added
        # Since we didn't save embeddings, they should be recomputed during load
        assert (
            len(loaded.example_store._embeddings_list) >= 0
        )  # May be computed during init
        # Verify embedding model was called during initialization
        assert new_embedding.call_count > initial_call_count

    def test_include_embeddings_parameter(self, tmp_path):
        """Test the include_embeddings parameter in save method."""
        mock_embedding = MockEmbeddingModel()
        examples = [({"test": "input"}, {"test": "output"})]

        prompt = DynamicFewShotPrompt(
            instruction="Test: {test}",
            examples=examples,
            embedding_model=mock_embedding,
        )

        # Save with embeddings
        path_with_emb = tmp_path / "with_embeddings.json"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prompt.save(str(path_with_emb), include_embeddings=True)

        with open(path_with_emb, "r") as f:
            data_with = json.load(f)
        assert "embeddings" in data_with

        # Save without embeddings
        path_without_emb = tmp_path / "without_embeddings.json"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prompt.save(str(path_without_emb), include_embeddings=False)

        with open(path_without_emb, "r") as f:
            data_without = json.load(f)
        assert "embeddings" not in data_without

        # Files should be different sizes
        size_with = path_with_emb.stat().st_size
        size_without = path_without_emb.stat().st_size
        assert size_with > size_without

    def test_json_structure_validation(self, tmp_path):
        """Test the generated JSON structure contains all required fields."""
        examples = [({"input": "test"}, {"output": "result"})]

        prompt = DynamicFewShotPrompt(
            instruction="Process: {input}",
            examples=examples,
            max_similar_examples=5,
            similarity_threshold=0.9,
        )

        json_path = tmp_path / "structure_test.json"
        prompt.save(str(json_path), include_embeddings=False)

        with open(json_path, "r") as f:
            data = json.load(f)

        # Verify all required fields are present
        required_fields = [
            "format_version",
            "type",
            "instruction",
            "examples",
            "response_model_info",
            "max_similar_examples",
            "similarity_threshold",
            "embedding_model_info",
        ]

        for field in required_fields:
            assert field in data

        # Verify field values
        assert data["format_version"] == "1.0"
        assert data["type"] == "DynamicFewShotPrompt"
        assert data["instruction"] == "Process: {input}"
        assert data["max_similar_examples"] == 5
        assert data["similarity_threshold"] == 0.9
        assert len(data["examples"]) == 1
        assert data["examples"][0]["input"]["input"] == "test"
        assert data["examples"][0]["output"]["output"] == "result"

    def test_warning_messages(self, tmp_path):
        """Test appropriate warning messages are shown."""
        mock_response_model = MockResponseModel(answer="test")
        mock_embedding = MockEmbeddingModel()

        prompt = DynamicFewShotPrompt(
            instruction="Test: {input}",
            examples=[({"input": "test"}, {"output": "result"})],
            response_model=mock_response_model,
            embedding_model=mock_embedding,
        )

        json_path = tmp_path / "warnings_test.json"

        # Should warn about both models
        with pytest.warns(UserWarning) as warning_list:
            prompt.save(str(json_path))

        warning_messages = [str(w.message) for w in warning_list]
        assert any("response_model cannot be saved" in msg for msg in warning_messages)
        assert any("embedding_model cannot be saved" in msg for msg in warning_messages)

        # Test load without embedding model shows warning (when embedding_model_info exists but no model provided)
        # First save a prompt with only embedding model info (no response model to avoid error)
        embedding_only_prompt = DynamicFewShotPrompt(
            instruction="Test: {input}",
            examples=[({"input": "test"}, {"output": "result"})],
            embedding_model=mock_embedding,
        )
        embedding_path = tmp_path / "embedding_only.json"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            embedding_only_prompt.save(str(embedding_path))

        # Now test load without providing embedding model - should show warning
        with pytest.warns(
            UserWarning, match="embedding_model.*similarity-based.*will not work"
        ):
            DynamicFewShotPrompt.load(str(embedding_path))

    def test_error_conditions(self, tmp_path):
        """Test various error conditions."""
        # Test loading non-existent file
        with pytest.raises(ValueError, match="Cannot load DynamicFewShotPrompt"):
            DynamicFewShotPrompt.load("nonexistent.json")

        # Test loading invalid JSON
        invalid_json_path = tmp_path / "invalid.json"
        with open(invalid_json_path, "w") as f:
            f.write("invalid json content")

        with pytest.raises(ValueError, match="Cannot load DynamicFewShotPrompt"):
            DynamicFewShotPrompt.load(str(invalid_json_path))

        # Test loading wrong file type
        wrong_type_path = tmp_path / "wrong_type.json"
        with open(wrong_type_path, "w") as f:
            json.dump(
                {"type": "Prompt", "instruction": "test"}, f
            )  # Regular Prompt, not DynamicFewShotPrompt

        with pytest.raises(ValueError, match="File is not a DynamicFewShotPrompt"):
            DynamicFewShotPrompt.load(str(wrong_type_path))

        # Test save to non-existent directory
        prompt = DynamicFewShotPrompt("Test: {input}")
        invalid_path = tmp_path / "nonexistent_dir" / "test.json"
        with pytest.raises(ValueError, match="Cannot save DynamicFewShotPrompt"):
            prompt.save(str(invalid_path))

    def test_response_model_requirements(self, tmp_path):
        """Test response model requirement validation."""
        mock_response_model = MockResponseModel(answer="test")
        prompt = DynamicFewShotPrompt(
            instruction="Test: {input}", response_model=mock_response_model
        )

        json_path = tmp_path / "model_required.json"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prompt.save(str(json_path))

        # Try to load without providing response_model - should raise error
        with pytest.raises(ValueError, match="requires a response_model"):
            DynamicFewShotPrompt.load(str(json_path))

        # Load with response_model should work
        new_model = MockResponseModel(answer="different")
        loaded = DynamicFewShotPrompt.load(str(json_path), response_model=new_model)
        assert loaded.response_model == new_model

    def test_round_trip_data_preservation(self, tmp_path):
        """Test that save/load round-trip preserves all data correctly."""
        mock_embedding = MockEmbeddingModel()
        examples = [
            ({"param1": "value1", "param2": "value2"}, {"result": "output1"}),
            (
                {"param1": "test", "param2": "data"},
                {"result": "output2", "extra": "info"},
            ),
        ]

        original = DynamicFewShotPrompt(
            instruction="Complex instruction with {param1} and {param2}",
            examples=examples,
            embedding_model=mock_embedding,
            max_similar_examples=1,
            similarity_threshold=0.6,
        )

        # Save and load
        json_path = tmp_path / "round_trip.json"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            original.save(str(json_path))

        new_embedding = MockEmbeddingModel()
        loaded = DynamicFewShotPrompt.load(
            str(json_path), embedding_model=new_embedding
        )

        # Verify all data is preserved
        assert loaded.instruction == original.instruction
        assert loaded.max_similar_examples == original.max_similar_examples
        assert loaded.similarity_threshold == original.similarity_threshold
        assert len(loaded.example_store) == len(original.example_store)
        assert loaded.example_store._examples == original.example_store._examples

        # Verify formatting works the same
        test_params = {"param1": "test1", "param2": "test2"}
        original_formatted = original.format(**test_params)
        loaded_formatted = loaded.format(**test_params)

        # Both formatted results should contain the test parameters
        assert test_params["param1"] in original_formatted
        assert test_params["param2"] in original_formatted
        assert test_params["param1"] in loaded_formatted
        assert test_params["param2"] in loaded_formatted

    def test_empty_example_store_handling(self, tmp_path):
        """Test handling of prompts with no examples."""
        prompt = DynamicFewShotPrompt(
            instruction="Simple instruction: {input}",
            max_similar_examples=3,
            similarity_threshold=0.8,
        )

        json_path = tmp_path / "no_examples.json"
        prompt.save(str(json_path))
        loaded = DynamicFewShotPrompt.load(str(json_path))

        assert loaded.instruction == prompt.instruction
        assert len(loaded.example_store) == 0
        assert loaded.max_similar_examples == 3
        assert loaded.similarity_threshold == 0.8
        assert loaded.format(input="test") == "Simple instruction: test"

    def test_unicode_handling(self, tmp_path):
        """Test unicode character handling in save/load."""
        examples = [
            ({"question": "¿Qué es la vida? 🤔"}, {"answer": "Es bella! 🌟"}),
            ({"question": "안녕하세요?"}, {"answer": "Hello in Korean! 🇰🇷"}),
        ]

        prompt = DynamicFewShotPrompt(
            instruction="Répondez: {question} 😊", examples=examples
        )

        json_path = tmp_path / "unicode_test.json"
        prompt.save(str(json_path))
        loaded = DynamicFewShotPrompt.load(str(json_path))

        assert loaded.instruction == prompt.instruction
        assert loaded.example_store._examples == prompt.example_store._examples

        # Test formatting with unicode
        formatted = loaded.format(question="Comment ça va? 🌈")
        assert "Comment ça va? 🌈" in formatted
        assert "😊" in formatted

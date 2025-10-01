import json
import tempfile
from pathlib import Path

import pytest

from ragas.metrics import DiscreteMetric, NumericMetric, RankingMetric
from ragas.prompt import DynamicFewShotPrompt, Prompt


class TestSimpleLLMMetricPersistence:
    """Test save and load functionality for SimpleLLMMetric and its subclasses."""

    def test_discrete_metric_save_and_load(self):
        """Test saving and loading a DiscreteMetric preserves all properties."""
        # Create metric with simple string prompt
        original_metric = DiscreteMetric(
            name="response_quality",
            prompt="Evaluate if the response '{response}' correctly answers the question '{question}'. Return 'correct' or 'incorrect'.",
            allowed_values=["correct", "incorrect"],
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            # Save to temp file
            original_metric.save(temp_path)

            # Verify file exists and is valid JSON
            assert Path(temp_path).exists()
            with open(temp_path, "r") as f:
                saved_data = json.load(f)

            # Basic structure checks
            assert saved_data["format_version"] == "1.0"
            assert saved_data["metric_type"] == "DiscreteMetric"
            assert saved_data["name"] == "response_quality"

            # Load from file
            loaded_metric = DiscreteMetric.load(temp_path)

            # Assert metric properties are identical
            assert loaded_metric.name == original_metric.name
            assert loaded_metric.allowed_values == original_metric.allowed_values
            assert (
                loaded_metric.prompt.instruction == original_metric.prompt.instruction
            )

            # Assert metric still functions (can score) - this will fail until we implement response_model handling
            # For now, just verify the basic properties

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_numeric_metric_save_and_load(self):
        """Test saving and loading a NumericMetric with range."""
        # Create metric with simple string prompt
        original_metric = NumericMetric(
            name="response_accuracy",
            prompt="Rate the accuracy of response '{response}' to question '{question}' on a scale of 0.0 to 1.0",
            allowed_values=(0.0, 1.0),
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            # Save to temp file
            original_metric.save(temp_path)

            # Load from file
            loaded_metric = NumericMetric.load(temp_path)

            # Assert metric properties are identical
            assert loaded_metric.name == original_metric.name
            assert loaded_metric.allowed_values == original_metric.allowed_values
            assert (
                loaded_metric.prompt.instruction == original_metric.prompt.instruction
            )

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_ranking_metric_save_and_load(self):
        """Test saving and loading a RankingMetric."""
        # Create metric with simple string prompt
        original_metric = RankingMetric(
            name="response_ranking",
            prompt="Rank these responses '{responses}' from best to worst for question '{question}'",
            allowed_values=5,  # Expected list length
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            # Save to temp file
            original_metric.save(temp_path)

            # Load from file
            loaded_metric = RankingMetric.load(temp_path)

            # Assert metric properties are identical
            assert loaded_metric.name == original_metric.name
            assert loaded_metric.allowed_values == original_metric.allowed_values
            assert (
                loaded_metric.prompt.instruction == original_metric.prompt.instruction
            )

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_save_load_with_prompt_object(self):
        """Test metric with Prompt object (not just string)."""
        # Create Prompt with examples
        prompt = Prompt(
            instruction="Evaluate if response '{response}' answers question '{question}'. Return 'good' or 'bad'.",
            examples=[
                (
                    {
                        "response": "The capital is Paris",
                        "question": "What is the capital of France?",
                    },
                    {"evaluation": "good"},
                ),
                (
                    {
                        "response": "I don't know",
                        "question": "What is the capital of France?",
                    },
                    {"evaluation": "bad"},
                ),
            ],
        )

        # Create metric with Prompt object
        original_metric = DiscreteMetric(
            name="response_evaluation", prompt=prompt, allowed_values=["good", "bad"]
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            # Save and load
            original_metric.save(temp_path)
            loaded_metric = DiscreteMetric.load(temp_path)

            # Verify prompt instruction and examples preserved
            assert (
                loaded_metric.prompt.instruction == original_metric.prompt.instruction
            )
            assert len(loaded_metric.prompt.examples) == len(
                original_metric.prompt.examples
            )

            # Verify examples content
            for orig_example, loaded_example in zip(
                original_metric.prompt.examples, loaded_metric.prompt.examples
            ):
                assert orig_example[0] == loaded_example[0]  # input
                assert orig_example[1] == loaded_example[1]  # output

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_save_load_with_dynamic_few_shot_prompt(self):
        """Test metric with DynamicFewShotPrompt."""

        # Create a mock embedding model for testing
        class MockEmbedding:
            def embed_query(self, text: str):
                # Simple mock - return hash-based embedding
                return [float(hash(text) % 1000) / 1000.0 for _ in range(10)]

            async def aembed_query(self, text: str):
                return self.embed_query(text)

        # Create DynamicFewShotPrompt
        base_prompt = Prompt("Evaluate response '{response}' for question '{question}'")
        embedding_model = MockEmbedding()

        dynamic_prompt = DynamicFewShotPrompt.from_prompt(
            base_prompt,
            embedding_model,
            max_similar_examples=3,
            similarity_threshold=0.7,
        )

        # Add some examples
        dynamic_prompt.add_example(
            {"response": "Good answer", "question": "Test question"},
            {"evaluation": "pass"},
        )

        # Create metric with DynamicFewShotPrompt
        original_metric = DiscreteMetric(
            name="dynamic_evaluation",
            prompt=dynamic_prompt,
            allowed_values=["pass", "fail"],
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            # Save (should warn about embedding model)
            with pytest.warns(UserWarning, match="embedding_model cannot be saved"):
                original_metric.save(temp_path)

            # Load (provide embedding model)
            loaded_metric = DiscreteMetric.load(
                temp_path, embedding_model=embedding_model
            )

            # Verify functionality - basic properties
            assert loaded_metric.name == original_metric.name
            assert loaded_metric.allowed_values == original_metric.allowed_values
            assert (
                loaded_metric.prompt.instruction == original_metric.prompt.instruction
            )
            assert (
                loaded_metric.prompt.max_similar_examples
                == original_metric.prompt.max_similar_examples
            )
            assert (
                loaded_metric.prompt.similarity_threshold
                == original_metric.prompt.similarity_threshold
            )

            # Verify examples were preserved
            assert len(loaded_metric.prompt.example_store._examples) == len(
                original_metric.prompt.example_store._examples
            )

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_save_with_default_path(self):
        """Test saving metric with default path uses metric name."""
        # Create metric
        original_metric = DiscreteMetric(
            name="test_default_save",
            prompt="Test prompt: {input}",
            allowed_values=["yes", "no"],
        )

        default_path = Path("test_default_save.json")

        try:
            # Save with no path argument - should use metric name
            original_metric.save()

            # Verify file was created with metric name
            assert default_path.exists()

            # Load and verify
            loaded_metric = DiscreteMetric.load(str(default_path))
            assert loaded_metric.name == original_metric.name
            assert (
                loaded_metric.prompt.instruction == original_metric.prompt.instruction
            )

        finally:
            default_path.unlink(missing_ok=True)

    def test_save_with_directory_path(self):
        """Test saving metric to a directory uses metric name as filename."""
        # Create metric
        original_metric = DiscreteMetric(
            name="test_dir_save",
            prompt="Test prompt: {input}",
            allowed_values=["yes", "no"],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            # Save to directory - should append metric name
            original_metric.save(temp_dir)

            expected_path = Path(temp_dir) / "test_dir_save.json"
            assert expected_path.exists()

            # Load and verify
            loaded_metric = DiscreteMetric.load(str(expected_path))
            assert loaded_metric.name == original_metric.name

    def test_save_with_no_extension(self):
        """Test saving metric without extension adds .json."""
        # Create metric
        original_metric = DiscreteMetric(
            name="test_no_ext",
            prompt="Test prompt: {input}",
            allowed_values=["yes", "no"],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir) / "my_metric"

            # Save without extension - should add .json
            original_metric.save(str(base_path))

            expected_path = base_path.with_suffix(".json")
            assert expected_path.exists()
            assert not base_path.exists()  # Should not create file without extension

            # Load and verify
            loaded_metric = DiscreteMetric.load(str(expected_path))
            assert loaded_metric.name == original_metric.name

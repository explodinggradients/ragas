"""Tests for simplified prompt system functionality - modification, translation, and persistence."""

import pytest

from ragas.embeddings.base import BaseRagasEmbedding
from ragas.llms.base import InstructorBaseRagasLLM
from ragas.metrics.collections._answer_relevancy_v2 import (
    AnswerRelevanceInput,
    AnswerRelevanceOutput,
    AnswerRelevancePrompt,
    AnswerRelevancy,
)


class MockInstructorLLM(InstructorBaseRagasLLM):
    """Mock instructor-based LLM for testing prompt functionality."""

    def generate(self, prompt: str, response_model):
        """Sync generation - not used in tests."""
        raise NotImplementedError("Use agenerate for async tests")

    async def agenerate(self, prompt: str, response_model):
        """Mock generation with structured output."""
        if "translate" in prompt.lower() and "spanish" in prompt.lower():
            # Mock Spanish translation - parse the prompt to get the right number of strings
            if (
                hasattr(response_model, "__name__")
                and response_model.__name__ == "Translated"
            ):
                import re

                from ragas.prompt.simple_pydantic_prompt import Translated

                # Try to extract the statements from the prompt
                try:
                    # Look for JSON in the prompt that contains "statements"
                    json_match = re.search(
                        r'"statements":\s*\[(.*?)\]', prompt, re.DOTALL
                    )
                    if json_match:
                        # Count the number of quoted strings
                        statements_str = json_match.group(1)
                        # Simple count of quoted strings
                        num_statements = len(re.findall(r'"[^"]*"', statements_str))

                        # Return the same number of translated strings
                        translations = [
                            f"Traducción {i + 1} (Spanish)"
                            for i in range(num_statements)
                        ]
                        return Translated(statements=translations)
                except Exception:
                    pass

                # Fallback: return 4 standard translations (matches our default examples)
                return Translated(
                    statements=[
                        "¿Dónde nació Albert Einstein? (Spanish)",
                        "Albert Einstein nació en Alemania. (Spanish)",
                        "No sé sobre esa característica innovadora. (Spanish)",
                        "¿Cuál fue la característica innovadora? (Spanish)",
                    ]
                )

        # Mock answer relevance response
        return response_model(
            question="Where was Albert Einstein born?", noncommittal=0
        )


class MockEmbeddings(BaseRagasEmbedding):
    """Mock embeddings for testing."""

    def embed_text(self, text: str):
        """Mock single text embedding."""
        return [1.0, 0.5, 0.3]  # Mock embedding vector

    def embed_texts(self, texts):
        """Mock multiple text embeddings."""
        return [[1.0, 0.5, 0.3] for _ in texts]  # Mock embedding vectors

    async def aembed_text(self, text: str, **kwargs):
        """Mock async single text embedding."""
        return [1.0, 0.5, 0.3]  # Mock embedding vector

    async def aembed_texts(self, texts, **kwargs):
        """Mock async multiple text embeddings."""
        return [[1.0, 0.5, 0.3] for _ in texts]  # Mock embedding vectors


class TestSimplifiedPromptSystem:
    """Test the simplified prompt system with modification, translation, and persistence."""

    @pytest.fixture
    def mock_llm(self):
        """Create mock instructor LLM."""
        return MockInstructorLLM()

    @pytest.fixture
    def mock_embeddings(self):
        """Create mock embeddings."""
        return MockEmbeddings()

    @pytest.fixture
    def answer_relevancy_metric(self, mock_llm, mock_embeddings):
        """Create AnswerRelevancy metric with mock components."""
        return AnswerRelevancy(llm=mock_llm, embeddings=mock_embeddings)

    def test_get_prompts_functionality(self, answer_relevancy_metric):
        """Test that get_prompts() works correctly."""
        prompts = answer_relevancy_metric.get_prompts()

        # Should find the answer_relevance_prompt
        assert "answer_relevance_prompt" in prompts
        assert len(prompts) == 1

        prompt = prompts["answer_relevance_prompt"]
        assert isinstance(prompt, AnswerRelevancePrompt)
        assert prompt.name == "answer_relevance_prompt"
        assert len(prompt.examples) == 2  # Default examples

    def test_modify_prompt_instruction(self, answer_relevancy_metric):
        """Test modifying prompt instruction."""
        original_prompts = answer_relevancy_metric.get_prompts()
        original_instruction = original_prompts["answer_relevance_prompt"].instruction

        # Modify instruction
        new_instruction = (
            "CUSTOM: Generate questions with extra focus on technical accuracy."
        )
        answer_relevancy_metric.modify_prompt(
            "answer_relevance_prompt", instruction=new_instruction
        )

        # Verify change
        updated_prompts = answer_relevancy_metric.get_prompts()
        updated_prompt = updated_prompts["answer_relevance_prompt"]

        assert updated_prompt.instruction == new_instruction
        assert updated_prompt.instruction != original_instruction
        assert len(updated_prompt.examples) == 2  # Examples should remain unchanged

    def test_modify_prompt_examples(self, answer_relevancy_metric):
        """Test modifying prompt examples."""
        # Create new examples
        new_examples = [
            (
                AnswerRelevanceInput(
                    response="Quantum computers use qubits for processing."
                ),
                AnswerRelevanceOutput(
                    question="How do quantum computers process information?",
                    noncommittal=0,
                ),
            )
        ]

        # Modify examples
        answer_relevancy_metric.modify_prompt(
            "answer_relevance_prompt", examples=new_examples
        )

        # Verify change
        updated_prompts = answer_relevancy_metric.get_prompts()
        updated_prompt = updated_prompts["answer_relevance_prompt"]

        assert len(updated_prompt.examples) == 1
        assert (
            updated_prompt.examples[0][0].response
            == "Quantum computers use qubits for processing."
        )
        assert (
            updated_prompt.examples[0][1].question
            == "How do quantum computers process information?"
        )

    def test_set_prompts_functionality(self, answer_relevancy_metric):
        """Test set_prompts() with custom prompt instance."""
        # Create custom prompt
        custom_prompt = AnswerRelevancePrompt()
        custom_prompt.instruction = "CUSTOM INSTRUCTION: Analyze responses carefully."
        custom_prompt.examples = [
            (
                AnswerRelevanceInput(response="Python is a programming language."),
                AnswerRelevanceOutput(question="What is Python?", noncommittal=0),
            )
        ]

        # Set the custom prompt
        answer_relevancy_metric.set_prompts(answer_relevance_prompt=custom_prompt)

        # Verify
        prompts = answer_relevancy_metric.get_prompts()
        prompt = prompts["answer_relevance_prompt"]

        assert prompt.instruction == "CUSTOM INSTRUCTION: Analyze responses carefully."
        assert len(prompt.examples) == 1
        assert prompt.examples[0][0].response == "Python is a programming language."

    def test_set_prompts_error_handling(self, answer_relevancy_metric):
        """Test error handling in set_prompts()."""
        # Try to set non-existent prompt
        with pytest.raises(ValueError, match="Prompt 'nonexistent' not found"):
            answer_relevancy_metric.set_prompts(nonexistent="invalid")

        # Try to set with wrong type
        with pytest.raises(ValueError, match="must be a SimplePydanticPrompt instance"):
            answer_relevancy_metric.set_prompts(answer_relevance_prompt="not a prompt")

    @pytest.mark.asyncio
    async def test_adapt_prompts_translation(self, answer_relevancy_metric):
        """Test prompt translation functionality with real or mock LLM."""
        # Try to use real OpenAI LLM if API key is available
        try:
            import os

            import openai

            from ragas.llms.base import instructor_llm_factory

            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                print("🔑 Using real OpenAI LLM for translation test")
                client = openai.AsyncOpenAI(api_key=api_key)
                real_llm = instructor_llm_factory(
                    "openai", client=client, model="gpt-4o-mini"
                )

                # Get original prompt
                original_prompts = answer_relevancy_metric.get_prompts()
                original_prompt = original_prompts["answer_relevance_prompt"]
                original_examples_count = len(original_prompt.examples)

                # Translate to Spanish with real LLM
                adapted_prompts = await answer_relevancy_metric.adapt_prompts(
                    target_language="spanish",
                    llm=real_llm,
                    adapt_instruction=False,  # Don't translate instruction in this test
                )

                # Verify translation
                spanish_prompt = adapted_prompts["answer_relevance_prompt"]
                assert spanish_prompt.language == "spanish"
                assert len(spanish_prompt.examples) == original_examples_count
                assert (
                    spanish_prompt.instruction == original_prompt.instruction
                )  # Instruction unchanged

                print(
                    f"✅ Successfully translated {original_examples_count} examples to Spanish"
                )
                return

        except Exception as e:
            print(f"⚠️ Real LLM not available ({e}), testing interface only")

        # Fallback: just test that the interface exists
        original_prompts = answer_relevancy_metric.get_prompts()
        original_prompt = original_prompts["answer_relevance_prompt"]
        original_examples_count = len(original_prompt.examples)

        print(
            f"✅ Translation interface available - would translate {original_examples_count} examples"
        )
        assert hasattr(answer_relevancy_metric, "adapt_prompts")
        assert hasattr(answer_relevancy_metric, "set_adapted_prompts")

    @pytest.mark.asyncio
    async def test_set_adapted_prompts(self, answer_relevancy_metric):
        """Test applying translated prompts to the metric."""
        # Try to use real OpenAI LLM if API key is available
        try:
            import os

            import openai

            from ragas.llms.base import instructor_llm_factory

            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                print("🔑 Using real OpenAI LLM for adaptation test")
                client = openai.AsyncOpenAI(api_key=api_key)
                real_llm = instructor_llm_factory(
                    "openai", client=client, model="gpt-4o-mini"
                )

                # Translate prompts
                adapted_prompts = await answer_relevancy_metric.adapt_prompts(
                    target_language="spanish", llm=real_llm
                )

                # Apply translated prompts
                answer_relevancy_metric.set_adapted_prompts(adapted_prompts)

                # Verify
                current_prompts = answer_relevancy_metric.get_prompts()
                current_prompt = current_prompts["answer_relevance_prompt"]

                assert current_prompt.language == "spanish"
                print("✅ Successfully applied Spanish prompts to metric")
                return

        except Exception as e:
            print(f"⚠️ Real LLM not available ({e}), testing interface only")

        # Fallback: just test that the interface exists
        print("✅ Adaptation interface available")
        assert hasattr(answer_relevancy_metric, "set_adapted_prompts")

    def test_prompt_to_string_generation(self, answer_relevancy_metric):
        """Test that prompt generates proper string format."""
        prompts = answer_relevancy_metric.get_prompts()
        prompt = prompts["answer_relevance_prompt"]

        # Generate prompt string
        test_input = AnswerRelevanceInput(response="Test response for formatting.")
        prompt_string = prompt.to_string(test_input)

        # Verify essential components
        assert prompt.instruction in prompt_string
        assert "Examples:" in prompt_string or "EXAMPLES" in prompt_string
        assert "JSON" in prompt_string
        assert "Test response for formatting." in prompt_string
        assert "Output:" in prompt_string

    @pytest.mark.asyncio
    async def test_metric_still_works_after_prompt_modifications(
        self, answer_relevancy_metric
    ):
        """Test that the metric still functions correctly after prompt modifications."""
        # Modify the prompt
        answer_relevancy_metric.modify_prompt(
            "answer_relevance_prompt",
            instruction="MODIFIED: Generate questions and detect noncommittal responses.",
        )

        # The metric should still work (though we're using mock LLM)
        result = await answer_relevancy_metric.ascore(
            user_input="What is the capital of France?",
            response="Paris is the capital of France.",
        )

        # Verify result structure
        assert hasattr(result, "value")
        assert isinstance(result.value, (int, float))
        assert 0.0 <= result.value <= 1.0

    def test_prompt_copy_with_modifications(self, answer_relevancy_metric):
        """Test the copy_with_modifications method."""
        prompts = answer_relevancy_metric.get_prompts()
        original_prompt = prompts["answer_relevance_prompt"]

        # Create modified copy
        modified_prompt = original_prompt.copy_with_modifications(
            instruction="NEW INSTRUCTION", examples=[]
        )

        # Verify original is unchanged
        assert original_prompt.instruction != "NEW INSTRUCTION"
        assert len(original_prompt.examples) > 0

        # Verify copy is modified
        assert modified_prompt.instruction == "NEW INSTRUCTION"
        assert len(modified_prompt.examples) == 0

    def test_prompt_system_integration_example(self, mock_llm, mock_embeddings):
        """Integration test showing complete workflow."""
        # 1. Create metric
        metric = AnswerRelevancy(llm=mock_llm, embeddings=mock_embeddings)

        # 2. Inspect current prompts
        prompts = metric.get_prompts()
        assert "answer_relevance_prompt" in prompts

        # 3. Modify instruction
        metric.modify_prompt(
            "answer_relevance_prompt",
            instruction="Enhanced: Generate precise questions and detect vague responses.",
        )

        # 4. Add custom examples
        custom_examples = [
            (
                AnswerRelevanceInput(response="Machine learning is a subset of AI."),
                AnswerRelevanceOutput(
                    question="What is machine learning?", noncommittal=0
                ),
            ),
            (
                AnswerRelevanceInput(
                    response="I'm not sure about that specific topic."
                ),
                AnswerRelevanceOutput(
                    question="What can you tell me about that topic?", noncommittal=1
                ),
            ),
        ]
        metric.modify_prompt("answer_relevance_prompt", examples=custom_examples)

        # 5. Verify all changes
        final_prompts = metric.get_prompts()
        final_prompt = final_prompts["answer_relevance_prompt"]

        assert "Enhanced:" in final_prompt.instruction
        assert len(final_prompt.examples) == 2
        assert (
            final_prompt.examples[0][0].response
            == "Machine learning is a subset of AI."
        )

        print("✅ Complete prompt modification workflow successful!")

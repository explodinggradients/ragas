"""End-to-end tests for the simplified prompt system using real OpenAI LLM."""

import os

import pytest

from ragas.metrics.collections._answer_relevancy_v2 import (
    AnswerRelevanceInput,
    AnswerRelevanceOutput,
    AnswerRelevancy,
)


class TestSimplifiedPromptSystemE2E:
    """End-to-end tests for simplified prompt system with real LLM."""

    @pytest.fixture
    def openai_api_key(self):
        """Get OpenAI API key from environment."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OPENAI_API_KEY not set - skipping E2E tests")
        return api_key

    @pytest.fixture
    def real_llm(self, openai_api_key):
        """Create real OpenAI instructor LLM."""
        try:
            import openai

            from ragas.llms.base import instructor_llm_factory

            client = openai.AsyncOpenAI(api_key=openai_api_key)
            return instructor_llm_factory("openai", client=client, model="gpt-4o-mini")
        except ImportError as e:
            pytest.skip(f"OpenAI not available: {e}")

    @pytest.fixture
    def real_embeddings(self, openai_api_key):
        """Create real OpenAI embeddings."""
        try:
            import openai

            from ragas.embeddings.base import embedding_factory

            client = openai.AsyncOpenAI(api_key=openai_api_key)
            return embedding_factory(
                provider="openai",
                model="text-embedding-ada-002",
                client=client,
                interface="modern",
            )
        except ImportError as e:
            pytest.skip(f"OpenAI embeddings not available: {e}")

    @pytest.fixture
    def answer_relevancy_metric(self, real_llm, real_embeddings):
        """Create AnswerRelevancy metric with real components."""
        return AnswerRelevancy(llm=real_llm, embeddings=real_embeddings)

    def test_get_prompts_with_real_metric(self, answer_relevancy_metric):
        """Test getting prompts from real metric."""
        prompts = answer_relevancy_metric.get_prompts()

        assert "answer_relevance_prompt" in prompts
        assert len(prompts) == 1

        prompt = prompts["answer_relevance_prompt"]
        assert prompt.name == "answer_relevance_prompt"
        assert len(prompt.examples) == 2  # Default examples

        print(f"✅ Found prompt with {len(prompt.examples)} examples")

    def test_modify_prompt_instruction_e2e(self, answer_relevancy_metric):
        """Test modifying prompt instruction end-to-end."""
        original_prompts = answer_relevancy_metric.get_prompts()
        original_instruction = original_prompts["answer_relevance_prompt"].instruction

        # Modify instruction
        new_instruction = "ENHANCED: Generate precise questions and detect vague responses with extra attention to technical accuracy."
        answer_relevancy_metric.modify_prompt(
            "answer_relevance_prompt", instruction=new_instruction
        )

        # Verify change
        updated_prompts = answer_relevancy_metric.get_prompts()
        updated_prompt = updated_prompts["answer_relevance_prompt"]

        assert updated_prompt.instruction == new_instruction
        assert updated_prompt.instruction != original_instruction

        print("✅ Successfully modified instruction")
        print(f"   Original: {original_instruction[:50]}...")
        print(f"   Modified: {new_instruction[:50]}...")

    def test_modify_prompt_examples_e2e(self, answer_relevancy_metric):
        """Test modifying prompt examples end-to-end."""
        # Create technical examples
        new_examples = [
            (
                AnswerRelevanceInput(
                    response="Machine learning algorithms can process large datasets to identify patterns."
                ),
                AnswerRelevanceOutput(
                    question="How do machine learning algorithms process datasets?",
                    noncommittal=0,
                ),
            ),
            (
                AnswerRelevanceInput(
                    response="I'm not entirely certain about the specific implementation details of that algorithm."
                ),
                AnswerRelevanceOutput(
                    question="What are the implementation details of that algorithm?",
                    noncommittal=1,
                ),
            ),
        ]

        # Modify examples
        answer_relevancy_metric.modify_prompt(
            "answer_relevance_prompt", examples=new_examples
        )

        # Verify change
        updated_prompts = answer_relevancy_metric.get_prompts()
        updated_prompt = updated_prompts["answer_relevance_prompt"]

        assert len(updated_prompt.examples) == 2
        assert "machine learning" in updated_prompt.examples[0][0].response.lower()
        assert updated_prompt.examples[1][1].noncommittal == 1

        print("✅ Successfully modified examples with technical content")

    @pytest.mark.asyncio
    async def test_real_translation_functionality(
        self, answer_relevancy_metric, real_llm
    ):
        """Test prompt translation with real OpenAI LLM."""
        print("\n🌍 Testing real translation functionality")

        # Get original prompt
        original_prompts = answer_relevancy_metric.get_prompts()
        original_prompt = original_prompts["answer_relevance_prompt"]
        original_examples_count = len(original_prompt.examples)

        print(f"   Original language: {original_prompt.language}")
        print(f"   Examples to translate: {original_examples_count}")

        # Translate to Spanish with real LLM
        adapted_prompts = await answer_relevancy_metric.adapt_prompts(
            target_language="spanish",
            llm=real_llm,
            adapt_instruction=True,  # Also translate instruction
        )

        # Verify translation
        spanish_prompt = adapted_prompts["answer_relevance_prompt"]

        assert spanish_prompt.language == "spanish"
        assert len(spanish_prompt.examples) == original_examples_count
        assert (
            spanish_prompt.instruction != original_prompt.instruction
        )  # Instruction translated

        print(f"   ✅ Translated language: {spanish_prompt.language}")
        print(f"   ✅ Examples preserved: {len(spanish_prompt.examples)}")
        print(f"   ✅ Instruction translated: {spanish_prompt.instruction[:50]}...")

        # Apply translated prompts
        answer_relevancy_metric.set_adapted_prompts(adapted_prompts)

        # Verify application
        current_prompts = answer_relevancy_metric.get_prompts()
        current_prompt = current_prompts["answer_relevance_prompt"]

        assert current_prompt.language == "spanish"
        print("   ✅ Spanish prompts successfully applied to metric")

    @pytest.mark.asyncio
    async def test_full_metric_functionality_after_modifications(
        self, answer_relevancy_metric, real_llm
    ):
        """Test that the metric works end-to-end after prompt modifications."""
        print("\n🧪 Testing full metric functionality after modifications")

        # 1. Modify the prompt
        answer_relevancy_metric.modify_prompt(
            "answer_relevance_prompt",
            instruction="CUSTOM: Generate questions and detect noncommittal responses with focus on technical topics.",
        )

        # 2. Test the metric still works
        result = await answer_relevancy_metric.ascore(
            user_input="What is the capital of France?",
            response="Paris is the capital of France, located in the north-central part of the country.",
        )

        # 3. Verify result
        assert hasattr(result, "value")
        assert isinstance(result.value, (int, float))
        assert 0.0 <= result.value <= 1.0

        print(f"   ✅ Metric score: {result.value:.4f}")
        print("   ✅ Score in valid range: [0.0, 1.0]")

        # 4. Test with noncommittal response
        noncommittal_result = await answer_relevancy_metric.ascore(
            user_input="What is quantum computing?",
            response="I'm not sure about the specific details of quantum computing.",
        )

        print(f"   ✅ Noncommittal score: {noncommittal_result.value:.4f}")

        # Noncommittal should generally score lower
        if noncommittal_result.value < result.value:
            print("   ✅ Noncommittal response correctly scored lower")
        else:
            print("   ⚠️ Noncommittal scoring may vary with LLM randomness")

    @pytest.mark.asyncio
    async def test_complete_workflow_e2e(self, real_llm, real_embeddings):
        """Complete end-to-end workflow test."""
        print("\n🚀 Testing complete simplified prompt system workflow")

        # 1. Create metric
        metric = AnswerRelevancy(llm=real_llm, embeddings=real_embeddings)
        print("   ✅ Created AnswerRelevancy metric")

        # 2. Inspect prompts
        prompts = metric.get_prompts()
        print(f"   ✅ Found {len(prompts)} prompt(s)")

        # 3. Modify instruction
        metric.modify_prompt(
            "answer_relevance_prompt",
            instruction="WORKFLOW TEST: Generate precise questions and identify evasive answers.",
        )
        print("   ✅ Modified instruction")

        # 4. Add custom examples
        custom_examples = [
            (
                AnswerRelevanceInput(
                    response="Artificial intelligence can solve complex problems."
                ),
                AnswerRelevanceOutput(
                    question="How does AI solve complex problems?", noncommittal=0
                ),
            ),
            (
                AnswerRelevanceInput(
                    response="I don't have enough information about that topic."
                ),
                AnswerRelevanceOutput(
                    question="What information is available about that topic?",
                    noncommittal=1,
                ),
            ),
        ]
        metric.modify_prompt("answer_relevance_prompt", examples=custom_examples)
        print("   ✅ Added custom examples")

        # 5. Test functionality
        result = await metric.ascore(
            user_input="How does machine learning work?",
            response="Machine learning uses algorithms to learn patterns from data and make predictions.",
        )
        print(f"   ✅ Metric evaluation successful: {result.value:.4f}")

        # 6. Translate to French
        adapted_prompts = await metric.adapt_prompts(
            "french", real_llm, adapt_instruction=True
        )
        metric.set_adapted_prompts(adapted_prompts)
        print("   ✅ Translated prompts to French")

        # 7. Test with French prompts
        french_result = await metric.ascore(
            user_input="Comment fonctionne l'apprentissage automatique?",
            response="L'apprentissage automatique utilise des algorithmes pour apprendre des modèles à partir de données.",
        )
        print(f"   ✅ French prompt evaluation: {french_result.value:.4f}")

        # 8. Verify final state
        final_prompts = metric.get_prompts()
        final_prompt = final_prompts["answer_relevance_prompt"]

        assert final_prompt.language == "french"
        assert len(final_prompt.examples) == 2
        assert (
            "WORKFLOW TEST" in final_prompt.instruction
            or "TEST DE FLUX" in final_prompt.instruction
        )

        print("   ✅ Final verification passed")
        print(
            f"   📊 Results: Original={result.value:.4f}, French={french_result.value:.4f}"
        )

        print("\n🎉 Complete workflow test successful!")
        print("   • Prompt discovery ✅")
        print("   • Instruction modification ✅")
        print("   • Example customization ✅")
        print("   • Metric functionality ✅")
        print("   • Translation ✅")
        print("   • End-to-end evaluation ✅")

    def test_prompt_system_documentation_compliance(self, answer_relevancy_metric):
        """Test that the simplified prompt system matches documentation examples."""
        print("\n📚 Testing documentation compliance")

        # Test get_prompts() like in docs
        prompts = answer_relevancy_metric.get_prompts()
        assert "answer_relevance_prompt" in prompts
        print("   ✅ get_prompts() works like documentation")

        # Test modify_prompt() like in docs
        answer_relevancy_metric.modify_prompt(
            "answer_relevance_prompt",
            instruction="Enhanced instruction from documentation example.",
        )

        updated_prompts = answer_relevancy_metric.get_prompts()
        assert (
            "Enhanced instruction"
            in updated_prompts["answer_relevance_prompt"].instruction
        )
        print("   ✅ modify_prompt() works like documentation")

        # Test set_prompts() interface
        custom_prompt = answer_relevancy_metric.get_prompts()["answer_relevance_prompt"]
        answer_relevancy_metric.set_prompts(answer_relevance_prompt=custom_prompt)
        print("   ✅ set_prompts() works like documentation")

        print("   📖 All documentation examples supported!")

    def test_error_handling_e2e(self, answer_relevancy_metric):
        """Test error handling in end-to-end scenarios."""
        print("\n🛡️ Testing error handling")

        # Test invalid prompt name
        with pytest.raises(ValueError, match="Prompt 'invalid' not found"):
            answer_relevancy_metric.modify_prompt("invalid", instruction="test")
        print("   ✅ Invalid prompt name properly rejected")

        # Test invalid prompt type
        with pytest.raises(ValueError, match="must be a SimplePydanticPrompt instance"):
            answer_relevancy_metric.set_prompts(answer_relevance_prompt="not a prompt")
        print("   ✅ Invalid prompt type properly rejected")

        print("   🛡️ Error handling working correctly!")

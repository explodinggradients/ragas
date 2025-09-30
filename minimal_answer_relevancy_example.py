"""
Transparent Routing Example: Legacy vs Modern LLMs with answer_relevancy metric.

This demonstrates the new transparent routing system where the same interface
automatically uses the best implementation based on the LLM type:
- Legacy LLMs → Question generation + embeddings approach
- Modern LLMs → Direct scoring approach (faster, no embeddings needed)

Callers use the same interface and never know which implementation is used!
"""

import asyncio
import os
from ragas.metrics import answer_relevancy
from ragas.dataset_schema import SingleTurnSample
from ragas.llms import llm_factory
from ragas.embeddings import embedding_factory


async def test_with_legacy_llm(question: str, answer: str, model: str = "gpt-4o-mini") -> float:
    """
    Test answer_relevancy with Legacy LLM (LangChain-based).
    
    Internally uses: Question generation + embeddings similarity approach
    
    Args:
        question: The user's question
        answer: The system's response/answer
        model: OpenAI model to use
        
    Returns:
        float: Answer relevancy score (0.0 to 1.0, higher is better)
    """
    # Legacy LLM - no client provided
    answer_relevancy.llm = llm_factory(model)  # No client = legacy
    answer_relevancy.embeddings = embedding_factory()  # Needed for legacy approach
    
    # Create a sample and evaluate - same interface!
    sample = SingleTurnSample(user_input=question, response=answer)
    score = await answer_relevancy._single_turn_ascore(sample, callbacks=[])
    return score


async def test_with_modern_llm(question: str, answer: str, provider: str = "openai", model: str = "gpt-4o-mini"):
    """
    Test answer_relevancy with Modern LLM (Instructor-based).
    
    Internally uses: Direct scoring approach (faster, single call)
    
    Args:
        question: The user's question
        answer: The system's response/answer
        provider: LLM provider (openai, anthropic, etc.)
        model: Model name to use
        
    Returns:
        float: Answer relevancy score (0.0 to 1.0, higher is better)
    """
    try:
        # Set up client based on provider
        if provider == "openai":
            from openai import AsyncOpenAI
            client = AsyncOpenAI()
        elif provider == "anthropic":
            import anthropic
            client = anthropic.AsyncAnthropic()
        else:
            raise ValueError(f"Provider {provider} not supported in this example")
        
        try:
            # Modern LLM - client provided signals modern intent
            answer_relevancy.llm = llm_factory(model, provider, client=client)
            # No need to set embeddings for modern approach!
            
            # Create a sample and evaluate - SAME INTERFACE!
            sample = SingleTurnSample(user_input=question, response=answer)
            score = await answer_relevancy._single_turn_ascore(sample, callbacks=[])
            return score
        finally:
            # Clean up async client to prevent cleanup warnings
            if hasattr(client, 'close'):
                await client.close()
        
    except ImportError as e:
        print(f"   ❌ {provider} client not available: {e}")
        return None
    except Exception as e:
        print(f"   ❌ Error with {provider}: {str(e)[:60]}...")
        return None


async def demonstrate_transparent_routing():
    """Demonstrate transparent routing - same interface, different implementations."""
    
    print("🔄 Transparent Routing: Legacy vs Modern LLMs")
    print("=" * 60)
    print("Same interface → Different implementations under the hood!")
    print()
    
    # Test cases
    test_cases = [
        ("What is Python?", "Python is a high-level programming language known for its simplicity and readability."),
        ("How do you make coffee?", "I don't know much about coffee preparation."),  # Non-committal
        ("What's the weather like?", "Python is a programming language."),  # Irrelevant
    ]
    
    for i, (question, answer) in enumerate(test_cases, 1):
        print(f"📝 Test Case {i}:")
        print(f"   Question: {question}")
        print(f"   Answer: {answer}")
        print()
        
        # Test Legacy LLM
        print("   🔗 Legacy LLM (LangChain):")
        print("      Implementation: Question generation + embeddings")
        try:
            legacy_score = await test_with_legacy_llm(question, answer)
            print(f"      Score: {legacy_score:.4f}")
            print(f"      Speed: Slower (multiple LLM calls + embeddings)")
        except Exception as e:
            print(f"      ❌ Error: {str(e)[:50]}...")
            legacy_score = None
        
        # Test Modern LLM
        print("   🎓 Modern LLM (Instructor):")
        print("      Implementation: Direct scoring")
        try:
            modern_score = await test_with_modern_llm(question, answer, "openai")
            print(f"      Score: {modern_score:.4f}")
            print(f"      Speed: Faster (single LLM call, no embeddings)")
        except Exception as e:
            print(f"      ❌ Error: {str(e)[:50]}...")
            modern_score = None
        
        # Compare scores
        if legacy_score is not None and modern_score is not None:
            diff = abs(legacy_score - modern_score)
            print(f"   📊 Score Difference: {diff:.4f}")
            if diff < 0.05:
                print(f"      → Very similar results ✅")
            elif diff < 0.15:
                print(f"      → Minor difference (expected)")
            else:
                print(f"      → Notable difference (different approaches)")
        
        print()


async def test_multiple_providers():
    """Test modern LLMs with multiple providers."""
    
    print("🌍 Testing Multiple Modern Providers")
    print("=" * 40)
    
    question = "What is machine learning?"
    answer = "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed."
    
    print(f"Question: {question}")
    print(f"Answer: {answer}")
    print()
    
    # Test different providers
    providers = [
        ("openai", "gpt-4o-mini"),
        ("anthropic", "claude-3-5-haiku-20241022"),  # Would need ANTHROPIC_API_KEY
    ]
    
    for provider, model in providers:
        print(f"🎓 Testing {provider.title()} ({model}):")
        
        # Check if API key is available
        api_key_var = f"{provider.upper()}_API_KEY"
        if not os.getenv(api_key_var) and provider != "openai":
            print(f"   ⚠️  {api_key_var} not set - skipping")
            continue
        
        try:
            score = await test_with_modern_llm(question, answer, provider, model)
            if score is not None:
                print(f"   ✅ Score: {score:.4f}")
                print(f"   ✅ Uses: Direct scoring (modern implementation)")
            else:
                print(f"   ❌ Failed to get score")
        except Exception as e:
            print(f"   ❌ Error: {str(e)[:50]}...")
        print()


async def demonstrate_same_interface():
    """Show that callers use the same interface regardless of LLM type."""
    
    print("🎯 Same Interface Demo")
    print("=" * 25)
    print("This function works with ANY LLM type!")
    print()
    
    async def evaluate_answer(llm, question: str, answer: str) -> float:
        """This function doesn't know or care about LLM type."""
        answer_relevancy.llm = llm
        if hasattr(llm, 'instructor_llm'):
            print("   🎓 Detected: Modern LLM (will use direct scoring)")
        else:
            print("   🔗 Detected: Legacy LLM (will use question generation)")
            answer_relevancy.embeddings = embedding_factory()
        
        sample = SingleTurnSample(user_input=question, response=answer)
        return await answer_relevancy._single_turn_ascore(sample, callbacks=[])
    
    question = "What is artificial intelligence?"
    answer = "AI is the simulation of human intelligence in machines."
    
    print(f"Question: {question}")
    print(f"Answer: {answer}")
    print()
    
    # Test with legacy LLM
    print("🔗 Testing with Legacy LLM:")
    legacy_llm = llm_factory("gpt-4o-mini")  # No client = legacy
    legacy_score = await evaluate_answer(legacy_llm, question, answer)
    print(f"   Score: {legacy_score:.4f}")
    print()
    
    # Test with modern LLM
    print("🎓 Testing with Modern LLM:")
    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI()
        try:
            modern_llm = llm_factory("gpt-4o-mini", client=client)  # Client = modern
            modern_score = await evaluate_answer(modern_llm, question, answer)
            print(f"   Score: {modern_score:.4f}")
        finally:
            await client.close()
    except ImportError:
        print("   ⚠️  OpenAI client not available")
    print()


def main():
    """Main function to run all demonstrations."""
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Please set OPENAI_API_KEY environment variable")
        return
    
    print("🚀 Answer Relevancy: Transparent Routing Demo")
    print("=" * 50)
    print("✅ Same interface works with both legacy and modern LLMs")
    print("✅ Automatic routing based on LLM type")
    print("✅ No breaking changes - existing code works unchanged")
    print("✅ Modern LLMs get performance benefits automatically")
    print()
    
    # Run demonstrations
    asyncio.run(demonstrate_transparent_routing())
    asyncio.run(test_multiple_providers())
    asyncio.run(demonstrate_same_interface())
    
    print("=" * 50)
    print("💡 Key Benefits:")
    print("   • ✅ Zero breaking changes - existing code works unchanged")
    print("   • ✅ Automatic optimization - modern LLMs use faster approach")
    print("   • ✅ Transparent routing - callers don't need to know LLM type")
    print("   • ✅ Same interface - answer_relevancy._single_turn_ascore() always works")
    print("   • ✅ Performance boost - modern path is faster (no embeddings needed)")
    print()
    print("🎯 Usage:")
    print("   # Legacy (existing code)")
    print("   answer_relevancy.llm = llm_factory('gpt-4o-mini')")
    print("   ")
    print("   # Modern (new capability)")
    print("   answer_relevancy.llm = llm_factory('gpt-4o-mini', client=OpenAI())")
    print("   ")
    print("   # Same interface for both!")
    print("   score = await answer_relevancy._single_turn_ascore(sample, callbacks=[])")


if __name__ == "__main__":
    main()

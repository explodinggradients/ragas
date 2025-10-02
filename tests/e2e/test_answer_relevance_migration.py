"""
Minimal example: Testing modern LLMs with answer_relevancy metric.

Demonstrates transparent routing where modern LLMs automatically use
faster direct scoring instead of question generation + embeddings.
"""

import asyncio
import os

from ragas.dataset_schema import SingleTurnSample
from ragas.llms import llm_factory
from ragas.metrics import answer_relevancy


async def demo_multiple_providers():
    """Test modern LLMs with multiple providers."""

    question = "What is machine learning?"
    answer = "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed."

    # Test different providers
    providers = [
        ("openai", "gpt-4o-mini"),
        ("anthropic", "claude-3-5-haiku-20241022"),
    ]

    print("Testing Modern LLM Providers:")
    print(f"Question: {question}")
    print(f"Answer: {answer}\n")

    for provider, model in providers:
        # Check if API key is available
        api_key_var = f"{provider.upper()}_API_KEY"
        if not os.getenv(api_key_var) and provider != "openai":
            print(f"{provider.title()}: API key not set - skipping")
            continue

        try:
            # Set up client
            if provider == "openai":
                from openai import AsyncOpenAI

                client = AsyncOpenAI()
            elif provider == "anthropic":
                import anthropic

                client = anthropic.AsyncAnthropic()
            else:
                continue

            try:
                # Modern LLM - client provided signals modern intent
                answer_relevancy.llm = llm_factory(model, provider, client=client)

                # Evaluate using same interface
                sample = SingleTurnSample(user_input=question, response=answer)
                score = await answer_relevancy._single_turn_ascore(sample, callbacks=[])

                print(f"{provider.title()}: Score {score:.4f} (direct scoring)")

            finally:
                if hasattr(client, "close"):
                    await client.close()

        except Exception as e:
            print(f"{provider.title()}: Error - {str(e)[:50]}...")


def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable")
        return

    asyncio.run(demo_multiple_providers())


if __name__ == "__main__":
    main()

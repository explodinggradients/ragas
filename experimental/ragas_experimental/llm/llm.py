__all__ = ["T", "RagasLLM", "ragas_llm"]

import asyncio
import inspect
import threading
import typing as t

import instructor
from pydantic import BaseModel

T = t.TypeVar("T", bound=BaseModel)


class RagasLLM:
    def __init__(self, provider: str, model: str, client: t.Any, **model_args):
        self.provider = provider.lower()
        self.model = model
        self.model_args = model_args or {}
        self.client = self._initialize_client(provider, client)
        # Check if client is async-capable at initialization
        self.is_async = self._check_client_async()

    def _check_client_async(self) -> bool:
        """Determine if the client is async-capable."""
        try:
            # Check if this is an async client by checking for a coroutine method
            if hasattr(self.client.chat.completions, "create"):
                return inspect.iscoroutinefunction(self.client.chat.completions.create)
            return False
        except (AttributeError, TypeError):
            return False

    def _initialize_client(self, provider: str, client: t.Any) -> t.Any:
        provider = provider.lower()

        if provider == "openai":
            return instructor.from_openai(client)
        elif provider == "anthropic":
            return instructor.from_anthropic(client)
        elif provider == "cohere":
            return instructor.from_cohere(client)
        elif provider == "gemini":
            return instructor.from_gemini(client)
        elif provider == "litellm":
            return instructor.from_litellm(client)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def _run_async_in_current_loop(self, coro):
        """Run an async coroutine in the current event loop if possible.

        This handles Jupyter environments correctly by using a separate thread
        when a running event loop is detected.
        """
        try:
            # Try to get the current event loop
            loop = asyncio.get_event_loop()

            if loop.is_running():
                # If the loop is already running (like in Jupyter notebooks),
                # we run the coroutine in a separate thread with its own event loop
                result_container = {"result": None, "exception": None}

                def run_in_thread():
                    # Create a new event loop for this thread
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        # Run the coroutine in this thread's event loop
                        result_container["result"] = new_loop.run_until_complete(coro)
                    except Exception as e:
                        # Capture any exceptions to re-raise in the main thread
                        result_container["exception"] = e
                    finally:
                        # Clean up the event loop
                        new_loop.close()

                # Start the thread and wait for it to complete
                thread = threading.Thread(target=run_in_thread)
                thread.start()
                thread.join()

                # Re-raise any exceptions that occurred in the thread
                if result_container["exception"]:
                    raise result_container["exception"]

                return result_container["result"]
            else:
                # Standard case - event loop exists but isn't running
                return loop.run_until_complete(coro)

        except RuntimeError:
            # If we get a runtime error about no event loop, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(coro)
            finally:
                # Clean up
                loop.close()
                asyncio.set_event_loop(None)

    def generate(self, prompt: str, response_model: t.Type[T]) -> T:
        """Generate a response using the configured LLM.

        For async clients, this will run the async method in the appropriate event loop.
        """
        messages = [{"role": "user", "content": prompt}]

        # If client is async, use the appropriate method to run it
        if self.is_async:
            return self._run_async_in_current_loop(
                self.agenerate(prompt, response_model)
            )
        else:
            # Regular sync client, just call the method directly
            return self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_model=response_model,
                **self.model_args,
            )

    async def agenerate(self, prompt: str, response_model: t.Type[T]) -> T:
        """Asynchronously generate a response using the configured LLM."""
        messages = [{"role": "user", "content": prompt}]

        # If client is not async, raise a helpful error
        if not self.is_async:
            raise TypeError(
                "Cannot use agenerate() with a synchronous client. Use generate() instead."
            )

        # Regular async client, call the method directly
        return await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_model=response_model,
            **self.model_args,
        )


def ragas_llm(provider: str, model: str, client: t.Any, **model_args) -> RagasLLM:
    return RagasLLM(provider=provider, client=client, model=model, **model_args)

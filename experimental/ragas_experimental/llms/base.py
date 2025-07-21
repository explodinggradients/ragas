__all__ = ["T", "InstructorLLM", "llm_factory", "BaseRagasLLM"]

import asyncio
from abc import ABC, abstractmethod
import inspect
import threading
import typing as t

import instructor
from pydantic import BaseModel

T = t.TypeVar("T", bound=BaseModel)


class BaseRagasLLM(ABC):
    @abstractmethod
    def generate(self, prompt: str, response_model: t.Type[T]) -> T:
        """Generate a response using the configured LLM.

        For async clients, this will run the async method in the appropriate event loop.
        """

    @abstractmethod
    async def agenerate(self, prompt: str, response_model: t.Type[T]) -> T:
        """Asynchronously generate a response using the configured LLM."""


class InstructorLLM(BaseRagasLLM):
    def __init__(self, client: t.Any, model: str, provider: str, **model_args):
        self.client = client
        self.model = model
        self.provider = provider
        self.model_args = model_args or {}
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

    def _get_client_info(self) -> str:
        """Get client type and async status information."""
        client_type = self.client.__class__.__name__
        async_status = "async" if self.is_async else "sync"
        return f"<{client_type}:{async_status}>"

    def _get_key_config(self) -> str:
        """Get key configuration parameters as a string."""
        config_parts = []

        # Show important model arguments
        important_args = [
            "temperature",
            "max_tokens",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
        ]
        for arg in important_args:
            if arg in self.model_args:
                config_parts.append(f"{arg}={self.model_args[arg]}")

        # Show count of other args if there are any
        other_args = len([k for k in self.model_args.keys() if k not in important_args])
        if other_args > 0:
            config_parts.append(f"+{other_args} more")

        return ", ".join(config_parts)

    def __repr__(self) -> str:
        """Return a detailed string representation of the LLM."""
        client_info = self._get_client_info()
        key_config = self._get_key_config()

        base_repr = f"InstructorLLM(provider='{self.provider}', model='{self.model}', client={client_info}"

        if key_config:
            base_repr += f", {key_config}"

        base_repr += ")"
        return base_repr

    __str__ = __repr__


def llm_factory(
    provider: str,
    model: t.Optional[str] = None,
    client: t.Optional[t.Any] = None,
    **kwargs: t.Any,
) -> BaseRagasLLM:
    """
    Factory function to create an LLM instance based on the provider.

    Args:
        provider (str): The name of the LLM provider or provider/model string
                       (e.g., "openai", "openai/gpt-4").
        model (str, optional): The model name to use for generation.
        client (Any, optional): Pre-initialized client for the provider.
        **kwargs: Additional arguments for the LLM (model_args).

    Returns:
        BaseRagasLLM: An instance of the specified LLM provider.

    Examples:
        # OpenAI with separate parameters
        llm = llm_factory("openai", "gpt-4", client=openai_client)

        # OpenAI with provider/model string
        llm = llm_factory("openai/gpt-4", client=openai_client)

        # Anthropic
        llm = llm_factory("anthropic", "claude-3-sonnet-20240229", client=anthropic_client)

        # Cohere
        llm = llm_factory("cohere", "command-r-plus", client=cohere_client)

        # Gemini
        llm = llm_factory("gemini", "gemini-pro", client=gemini_client)

        # LiteLLM (supports 100+ models)
        llm = llm_factory("litellm", "gpt-4", client=litellm_client)

    Raises:
        ValueError: If provider is unsupported or required parameters are missing.
    """
    # Handle provider/model string format
    if "/" in provider and model is None:
        provider_name, model_name = provider.split("/", 1)
        provider = provider_name
        model = model_name

    if not model:
        raise ValueError(
            "Model name is required. Either provide it as a separate parameter "
            "or use provider/model format (e.g., 'openai/gpt-4')"
        )

    def _initialize_client(provider: str, client: t.Any) -> t.Any:
        """Initialize the instructor-patched client for the given provider."""
        if not client:
            raise ValueError(f"{provider.title()} provider requires a client instance")

        provider_lower = provider.lower()

        if provider_lower == "openai":
            return instructor.from_openai(client)
        elif provider_lower == "anthropic":
            return instructor.from_anthropic(client)
        elif provider_lower == "cohere":
            return instructor.from_cohere(client)
        elif provider_lower == "gemini":
            return instructor.from_gemini(client)
        elif provider_lower == "litellm":
            return instructor.from_litellm(client)
        else:
            raise ValueError(
                f"Unsupported provider: {provider}. "
                f"Supported providers: openai, anthropic, cohere, gemini, litellm"
            )

    instructor_patched_client = _initialize_client(provider=provider, client=client)
    return InstructorLLM(
        client=instructor_patched_client, model=model, provider=provider, **kwargs
    )

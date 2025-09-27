"""OCI Gen AI LLM wrapper implementation for Ragas."""

import asyncio
import logging
import typing as t
from dataclasses import dataclass

from langchain_core.outputs import LLMResult, Generation
from langchain_core.prompt_values import PromptValue

from ragas._analytics import LLMUsageEvent, track
from ragas.exceptions import LLMDidNotFinishException
from ragas.llms.base import BaseRagasLLM
from ragas.run_config import RunConfig

logger = logging.getLogger(__name__)


class OCIGenAIWrapper(BaseRagasLLM):
    """
    OCI Gen AI LLM wrapper for Ragas.
    
    This wrapper provides direct integration with Oracle Cloud Infrastructure
    Generative AI services without requiring LangChain or LlamaIndex.
    """

    def __init__(
        self,
        model_id: str,
        compartment_id: str,
        config: t.Optional[t.Dict[str, t.Any]] = None,
        endpoint_id: t.Optional[str] = None,
        run_config: t.Optional[RunConfig] = None,
        cache: t.Optional[t.Any] = None,
    ):
        """
        Initialize OCI Gen AI wrapper.
        
        Args:
            model_id: The OCI model ID to use for generation
            compartment_id: The OCI compartment ID
            config: OCI configuration dictionary (optional, uses default if not provided)
            endpoint_id: Optional endpoint ID for the model
            run_config: Ragas run configuration
            cache: Optional cache backend
        """
        super().__init__(cache=cache)
        
        # Import OCI SDK
        try:
            import oci
            from oci.generative_ai import GenerativeAiClient
        except ImportError:
            raise ImportError(
                "OCI SDK not found. Please install it with: pip install oci"
            )
        
        self.model_id = model_id
        self.compartment_id = compartment_id
        self.endpoint_id = endpoint_id
        
        # Initialize OCI client
        if config is None:
            config = oci.config.from_file()
        
        self.client = GenerativeAiClient(config)
        
        # Set run config
        if run_config is None:
            run_config = RunConfig()
        self.set_run_config(run_config)
        
        # Track initialization
        track(
            LLMUsageEvent(
                provider="oci_genai",
                model=model_id,
                llm_type="oci_wrapper",
                num_requests=1,
                is_async=False,
            )
        )

    def _convert_prompt_to_string(self, prompt: PromptValue) -> str:
        """Convert PromptValue to string."""
        if hasattr(prompt, 'to_string'):
            return prompt.to_string()
        elif hasattr(prompt, 'to_messages'):
            messages = prompt.to_messages()
            return "\n".join([msg.content for msg in messages if hasattr(msg, 'content')])
        else:
            return str(prompt)

    def _create_generation_request(
        self,
        prompt: str,
        temperature: float = 0.01,
        max_tokens: t.Optional[int] = None,
        stop: t.Optional[t.List[str]] = None,
    ) -> t.Dict[str, t.Any]:
        """Create generation request for OCI Gen AI."""
        request = {
            "compartment_id": self.compartment_id,
            "serving_mode": {
                "model_id": self.model_id
            },
            "inference_request": {
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": max_tokens or 1000,
                "temperature": temperature,
            }
        }
        
        if self.endpoint_id:
            request["serving_mode"] = {"endpoint_id": self.endpoint_id}
        
        if stop:
            request["inference_request"]["stop"] = stop
            
        return request

    def generate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: t.Optional[float] = 0.01,
        stop: t.Optional[t.List[str]] = None,
        callbacks: t.Optional[t.Any] = None,
    ) -> LLMResult:
        """Generate text using OCI Gen AI."""
        if temperature is None:
            temperature = self.get_temperature(n)
        
        prompt_str = self._convert_prompt_to_string(prompt)
        generations = []
        
        try:
            for _ in range(n):
                request = self._create_generation_request(
                    prompt_str, temperature, stop=stop
                )
                
                response = self.client.generate_text(**request)
                
                # Extract text from response
                if hasattr(response.data, 'choices') and response.data.choices:
                    text = response.data.choices[0].message.content
                elif hasattr(response.data, 'text'):
                    text = response.data.text
                else:
                    text = str(response.data)
                
                generation = Generation(text=text)
                generations.append([generation])
            
            # Track usage
            track(
                LLMUsageEvent(
                    provider="oci_genai",
                    model=self.model_id,
                    llm_type="oci_wrapper",
                    num_requests=n,
                    is_async=False,
                )
            )
            
            return LLMResult(generations=generations)
            
        except Exception as e:
            logger.error(f"Error generating text with OCI Gen AI: {e}")
            raise

    async def agenerate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: t.Optional[float] = 0.01,
        stop: t.Optional[t.List[str]] = None,
        callbacks: t.Optional[t.Any] = None,
    ) -> LLMResult:
        """Generate text asynchronously using OCI Gen AI."""
        if temperature is None:
            temperature = self.get_temperature(n)
        
        prompt_str = self._convert_prompt_to_string(prompt)
        generations = []
        
        try:
            # Run synchronous calls in thread pool for async compatibility
            loop = asyncio.get_event_loop()
            
            for _ in range(n):
                request = self._create_generation_request(
                    prompt_str, temperature, stop=stop
                )
                
                response = await loop.run_in_executor(
                    None, lambda: self.client.generate_text(**request)
                )
                
                # Extract text from response
                if hasattr(response.data, 'choices') and response.data.choices:
                    text = response.data.choices[0].message.content
                elif hasattr(response.data, 'text'):
                    text = response.data.text
                else:
                    text = str(response.data)
                
                generation = Generation(text=text)
                generations.append([generation])
            
            # Track usage
            track(
                LLMUsageEvent(
                    provider="oci_genai",
                    model=self.model_id,
                    llm_type="oci_wrapper",
                    num_requests=n,
                    is_async=True,
                )
            )
            
            return LLMResult(generations=generations)
            
        except Exception as e:
            logger.error(f"Error generating text with OCI Gen AI: {e}")
            raise

    def is_finished(self, response: LLMResult) -> bool:
        """Check if the LLM response is finished/complete."""
        # For OCI Gen AI, we assume the response is always finished
        # unless there's an explicit error or truncation
        try:
            for generation_list in response.generations:
                for generation in generation_list:
                    if not generation.text or generation.text.strip() == "":
                        return False
            return True
        except Exception:
            return False

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_id={self.model_id}, compartment_id={self.compartment_id})"


def oci_genai_factory(
    model_id: str,
    compartment_id: str,
    config: t.Optional[t.Dict[str, t.Any]] = None,
    endpoint_id: t.Optional[str] = None,
    run_config: t.Optional[RunConfig] = None,
    **kwargs: t.Any,
) -> OCIGenAIWrapper:
    """
    Factory function to create an OCI Gen AI LLM instance.
    
    Args:
        model_id: The OCI model ID to use for generation
        compartment_id: The OCI compartment ID
        config: OCI configuration dictionary (optional)
        endpoint_id: Optional endpoint ID for the model
        run_config: Ragas run configuration
        **kwargs: Additional arguments passed to OCIGenAIWrapper
        
    Returns:
        OCIGenAIWrapper: An instance of the OCI Gen AI LLM wrapper
        
    Examples:
        # Basic usage with default config
        llm = oci_genai_factory(
            model_id="cohere.command",
            compartment_id="ocid1.compartment.oc1..example"
        )
        
        # With custom config
        llm = oci_genai_factory(
            model_id="cohere.command",
            compartment_id="ocid1.compartment.oc1..example",
            config={"user": "user_ocid", "key_file": "~/.oci/private_key.pem"}
        )
    """
    return OCIGenAIWrapper(
        model_id=model_id,
        compartment_id=compartment_id,
        config=config,
        endpoint_id=endpoint_id,
        run_config=run_config,
        **kwargs
    )

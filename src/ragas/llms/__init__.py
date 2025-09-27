from ragas.llms.base import (
    BaseRagasLLM,
    InstructorBaseRagasLLM,
    InstructorLLM,
    InstructorTypeVar,
    LangchainLLMWrapper as _LangchainLLMWrapper,
    LlamaIndexLLMWrapper as _LlamaIndexLLMWrapper,
    instructor_llm_factory,
    llm_factory,
)
from ragas.llms.batch_api import (
    BatchEndpoint,
    BatchJob,
    BatchRequest,
    BatchResponse,
    BatchStatus,
    OpenAIBatchAPI,
    create_batch_api,
)
from ragas.llms.haystack_wrapper import HaystackLLMWrapper
from ragas.utils import DeprecationHelper

# Create deprecation wrappers for legacy classes
LangchainLLMWrapper = DeprecationHelper(
    _LangchainLLMWrapper,
    "LangchainLLMWrapper is deprecated and will be removed in a future version. "
    "Use the modern LLM providers instead: "
    "from ragas.llms.base import llm_factory; llm = llm_factory('gpt-4o-mini') "
    "or from ragas.llms.base import instructor_llm_factory; llm = instructor_llm_factory('openai', client=openai_client)",
)

LlamaIndexLLMWrapper = DeprecationHelper(
    _LlamaIndexLLMWrapper,
    "LlamaIndexLLMWrapper is deprecated and will be removed in a future version. "
    "Use the modern LLM providers instead: "
    "from ragas.llms.base import llm_factory; llm = llm_factory('gpt-4o-mini') "
    "or from ragas.llms.base import instructor_llm_factory; llm = instructor_llm_factory('openai', client=openai_client)",
)

__all__ = [
    "BaseRagasLLM",
    "HaystackLLMWrapper",
    "InstructorBaseRagasLLM",
    "InstructorLLM",
    "LangchainLLMWrapper",
    "LlamaIndexLLMWrapper",
    "InstructorTypeVar",
    "instructor_llm_factory",
    "llm_factory",
    # Batch API
    "BatchEndpoint",
    "BatchJob",
    "BatchRequest",
    "BatchResponse",
    "BatchStatus",
    "OpenAIBatchAPI",
    "create_batch_api",
]

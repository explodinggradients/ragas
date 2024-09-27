import typing as t
from dataclasses import dataclass, field

from ragas.experimental.testset.generators.base import BaseTestsetGenerator
from ragas.experimental.testset.graph import KnowledgeGraph
from ragas.experimental.testset.transforms import BaseGraphTransformations, Parallel
from ragas.llms import BaseRagasLLM, LangchainLLMWrapper
from ragas.run_config import RunConfig

if t.TYPE_CHECKING:
    from langchain_core.documents import Document as LCDocument
    from langchain_core.language_models import BaseLanguageModel as LangchainLLM
    from llama_index.core.base.llms.base import BaseLLM as LlamaindexLLM
    from llama_index.core.schema import Document as LlamaindexDocument

QuestionTypes = t.Dict[BaseTestsetGenerator, float]


@dataclass
class TestsetGenerator:
    llm: BaseRagasLLM
    docstore: KnowledgeGraph = field(default_factory=KnowledgeGraph)

    @classmethod
    def from_langchain(
        cls,
        llm: LangchainLLM,
        docstore: t.Optional[KnowledgeGraph] = None,
    ):
        docstore = docstore or KnowledgeGraph()
        transforms = transforms or []

        wrapped_generator_llm = LangchainLLMWrapper(generator_llm)
        wrapped_critic_llm = LangchainLLMWrapper(critic_llm)
        return cls(wrapped_generator_llm, wrapped_critic_llm, docstore, transforms)

    def generate_with_langchain_docs(
        self,
        documents: t.Sequence[LCDocument],
        test_size: int,
        scenarios: t.Optional[QuestionTypes] = None,
        with_debugging_logs=False,
        is_async: bool = True,
        raise_exceptions: bool = True,
        run_config: t.Optional[RunConfig] = None,
    ):
        pass

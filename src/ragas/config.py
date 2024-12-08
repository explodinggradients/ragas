import typing as t

from pydantic import BaseModel, Field

from ragas.embeddings import BaseRagasEmbeddings, embedding_factory
from ragas.llms import BaseRagasLLM
from ragas.losses import Loss
from ragas.optimizers import GeneticOptimizer, Optimizer

DEFAULT_OPTIMIZER_CONFIG = {"max_steps": 100}


class DemonstrationConfig(BaseModel):
    enabled: bool = True
    top_k: int = 3
    technique: t.Literal["random", "similarity"] = "similarity"
    embedding: BaseRagasEmbeddings = Field(default_factory=lambda: embedding_factory())


class InstructionConfig(BaseModel):
    enabled: bool = True
    loss: t.Optional[Loss] = None
    optimizer: Optimizer = GeneticOptimizer()
    optimizer_config: t.Dict[str, t.Any] = Field(
        default_factory=lambda: DEFAULT_OPTIMIZER_CONFIG
    )
    llm: t.Optional[BaseRagasLLM] = None

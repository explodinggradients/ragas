import typing as t

from pydantic import BaseModel, Field

from ragas.embeddings import BaseRagasEmbeddings
from ragas.llms import BaseRagasLLM
from ragas.optimizers.base import Optimizer

DEFAULT_OPTIMIZER_CONFIG = {"max_steps": 100}


class DemonstrationConfig(BaseModel):
    enabled: bool = True
    top_k: int = 3
    technique: t.Literal["random", "similarity"] = "similarity"
    embedding: t.Optional[BaseRagasEmbeddings] = None


class InstructionConfig(BaseModel):
    enabled: bool = True
    optimizer: Optimizer
    optimizer_config: t.Dict[str, t.Any] = Field(
        default_factory=lambda: DEFAULT_OPTIMIZER_CONFIG
    )
    model: t.Optional[BaseRagasLLM] = None

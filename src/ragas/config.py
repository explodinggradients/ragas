from __future__ import annotations

import typing as t

from pydantic import BaseModel, Field

from ragas.embeddings import BaseRagasEmbeddings, embedding_factory
from ragas.llms import BaseRagasLLM, llm_factory
from ragas.losses import Loss
from ragas.optimizers import GeneticOptimizer, Optimizer

DEFAULT_OPTIMIZER_CONFIG = {"max_steps": 100}


class DemonstrationConfig(BaseModel):
    embedding: BaseRagasEmbeddings = Field(default_factory=embedding_factory)
    enabled: bool = True
    top_k: int = 3
    technique: t.Literal["random", "similarity"] = "similarity"


class InstructionConfig(BaseModel):
    llm: BaseRagasLLM = Field(default_factory=llm_factory)
    enabled: bool = True
    loss: t.Optional[Loss] = None
    optimizer: Optimizer = GeneticOptimizer()
    optimizer_config: t.Dict[str, t.Any] = Field(
        default_factory=lambda: DEFAULT_OPTIMIZER_CONFIG
    )


InstructionConfig.model_rebuild()

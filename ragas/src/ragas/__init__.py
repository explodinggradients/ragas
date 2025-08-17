from ragas.cache import CacheInterface, DiskCacheBackend, cacher
from ragas.dataset_schema import EvaluationDataset, MultiTurnSample, SingleTurnSample
from ragas.evaluation import evaluate
from ragas.run_config import RunConfig
from ragas.simulation import (
    Message,
    ConversationHistory,
    UserSimulator,
    UserSimulatorResponse,
    validate_agent_function,
    validate_stopping_criteria,
    default_stopping_criteria,
)

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown version"


__all__ = [
    "evaluate",
    "RunConfig",
    "__version__",
    "SingleTurnSample",
    "MultiTurnSample",
    "EvaluationDataset",
    "cacher",
    "CacheInterface",
    "DiskCacheBackend",
    "Message",
    "ConversationHistory",
    "UserSimulator",
    "UserSimulatorResponse",
    "validate_agent_function",
    "validate_stopping_criteria",
    "default_stopping_criteria",
]


def __getattr__(name):
    if name == "experimental":
        try:
            import ragas_experimental as experimental  # type: ignore

            return experimental
        except ImportError:
            raise ImportError(
                "ragas.experimental requires installation: "
                "pip install ragas[experimental]"
            )
    raise AttributeError(f"module 'ragas' has no attribute '{name}'")

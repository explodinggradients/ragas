# Backend imports
# Backend imports
from ragas import backends
from ragas.cache import CacheInterface, DiskCacheBackend, cacher
from ragas.dataset_schema import EvaluationDataset, MultiTurnSample, SingleTurnSample
from ragas.evaluation import evaluate
from ragas.experiment import Experiment, experiment, version_experiment
from ragas.run_config import RunConfig
from ragas.simulation import (
    ConversationHistory,
    Message,
    UserSimulator,
    UserSimulatorResponse,
    default_stopping_criteria,
    validate_agent_function,
    validate_stopping_criteria,
)

# Backend imports
from ragas import backends

# Backend imports

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
    "backends",
    "Experiment",
    "experiment",
    "version_experiment",
    "ConversationHistory",
    "Message",
    "UserSimulator",
    "UserSimulatorResponse",
    "default_stopping_criteria",
    "validate_agent_function",
    "validate_stopping_criteria",
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

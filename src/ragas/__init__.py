from ragas.adaptation import adapt
from ragas.evaluation import evaluate
from ragas.run_config import RunConfig

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown version"


__all__ = ["evaluate", "adapt", "RunConfig", "__version__"]

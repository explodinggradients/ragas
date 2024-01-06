from ragas.adaptation import adapt
from ragas.evaluation import evaluate

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown version"


__all__ = ["evaluate", "adapt", "__version__"]

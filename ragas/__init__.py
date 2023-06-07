from ragas.evaluation import evaluate

try:
    from ._version import version as __version__
    from ._version import version_tuple
except ImportError:
    __version__ = "unknown version"
    version_tuple = (0, 0, "unknown version")


__all__ = ["evaluate", "__version__", "version_tuple"]

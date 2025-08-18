# Get version from setuptools_scm-generated file
try:
    from ._version import version as __version__  # type: ignore
except ImportError:
    # Fallback for installed package
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as pkg_version

    try:
        __version__ = pkg_version("ragas")
    except PackageNotFoundError:
        __version__ = "unknown"

from .dataset import Dataset
from .llms import llm_factory
from .embeddings import embedding_factory

__all__ = ["Dataset", "llm_factory", "embedding_factory"]

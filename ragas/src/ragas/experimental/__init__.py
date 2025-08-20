# Get version from setuptools_scm-generated file
try:
    from ._version import version as __version__  # type: ignore
except ImportError:
    # Fallback for installed package
    from importlib.metadata import PackageNotFoundError, version as pkg_version

    try:
        __version__ = pkg_version("ragas")
    except PackageNotFoundError:
        __version__ = "unknown"

from ragas.embeddings import embedding_factory
from ragas.llms import llm_factory

__all__ = ["embedding_factory", "llm_factory"]

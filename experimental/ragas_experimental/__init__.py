# Get version from setuptools_scm-generated file
try:
    from ._version import version as __version__
except ImportError:
    # Fallback for installed package
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as pkg_version

    try:
        __version__ = pkg_version("ragas_experimental")
    except PackageNotFoundError:
        __version__ = "unknown"

from ragas_experimental.dataset import Dataset
from ragas_experimental.experiment import experiment, Experiment

__all__ = ["Dataset", "experiment", "Experiment"]

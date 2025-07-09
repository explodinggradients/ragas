"""Local CSV backend implementation for projects and datasets."""

from .base import BaseBackend


class LocalCSVBackend(BaseBackend):
    """Local CSV implementation of DataTableBackend."""

    def __init__(
        self,
        root_dir: str,
    ):
        self.root_dir = root_dir

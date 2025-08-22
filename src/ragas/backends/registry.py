"""Backend registry for managing and discovering project backends."""

import logging
import typing as t
from importlib import metadata

from .base import BaseBackend

logger = logging.getLogger(__name__)


class BackendRegistry:
    """Registry for managing project backends with plugin support."""

    _instance = None
    _backends: t.Dict[str, t.Type[BaseBackend]] = {}
    _aliases: t.Dict[str, str] = {}
    _discovered = False

    def __new__(cls):
        """Singleton pattern to ensure single registry instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _resolve_name(self, name: str) -> str:
        """Resolve alias to primary name, return name if not an alias."""
        return self._aliases.get(name, name)

    def _get_available_names(self) -> t.List[str]:
        """Get list of all available names (primary names + aliases) for error messages."""
        if not self._discovered:
            self.discover_backends()
        return list(self._backends.keys()) + list(self._aliases.keys())

    def _get_aliases_for(self, primary_name: str) -> t.List[str]:
        """Get all aliases pointing to a primary backend name."""
        return [
            alias for alias, target in self._aliases.items() if target == primary_name
        ]

    def _validate_name(self, name: str) -> None:
        """Validate backend name format."""
        if not name or not isinstance(name, str):
            raise ValueError("Backend name must be a non-empty string")

    def _validate_backend_class(self, backend_class: t.Type[BaseBackend]) -> None:
        """Validate backend class inheritance."""
        if not issubclass(backend_class, BaseBackend):
            raise TypeError(
                f"Backend class {backend_class} must inherit from BaseBackend"
            )

    def register_aliases(
        self, name: str, aliases: t.List[str], overwrite: bool = False
    ) -> None:
        """Register aliases for an existing backend.

        Args:
            name: Primary name of the backend
            aliases: List of alternative names for the backend
            overwrite: Whether to overwrite existing aliases

        Raises:
            KeyError: If backend name doesn't exist
        """
        if name not in self._backends:
            raise KeyError(f"Backend '{name}' not found")

        for alias in aliases:
            if not alias or not isinstance(alias, str):
                logger.warning(
                    f"Invalid alias '{alias}' for backend '{name}', skipping"
                )
                continue

            if alias in self._aliases and not overwrite:
                logger.warning(f"Alias '{alias}' already exists, skipping")
                continue

            self._aliases[alias] = name
            logger.debug(f"Registered backend alias: {alias} -> {name}")

    def list_all_names(self) -> t.Dict[str, t.List[str]]:
        """List all backend names including aliases.

        Returns:
            Dictionary mapping primary names to lists of all names (including aliases)
        """
        if not self._discovered:
            self.discover_backends()
        return {
            primary_name: [primary_name] + self._get_aliases_for(primary_name)
            for primary_name in self._backends.keys()
        }

    def discover_backends(self) -> t.Dict[str, t.Type[BaseBackend]]:
        """Discover and register backends from entry points.

        Returns:
            Dictionary of discovered backends
        """
        if self._discovered:
            return self._backends.copy()

        self._discover_backends()
        self._discovered = True
        logger.info(f"Discovered {len(self._backends)} backends from entry points.")

        return self._backends.copy()

    def _discover_backends(self) -> None:
        """Discover backends from setuptools entry points."""
        try:
            entry_points_result = metadata.entry_points()

            # Python 3.10+ has .select() method, Python 3.9 returns a dict
            if hasattr(entry_points_result, "select"):
                # Python 3.10+
                entry_points = entry_points_result.select(group="ragas.backends")  # type: ignore[attr-defined]
            else:
                # Python 3.9 compatibility
                entry_points = (
                    entry_points_result.get("ragas.backends", [])
                    if isinstance(entry_points_result, dict)
                    else []
                )

            for entry_point in entry_points:
                try:
                    self[entry_point.name] = entry_point.load()
                    logger.debug(f"Loaded backend: {entry_point.name}")
                except Exception as e:
                    logger.warning(f"Failed to load backend '{entry_point.name}': {e}")
        except Exception as e:
            logger.debug(f"No entry points found: {e}")

    def get_backend_info(self, name: str) -> t.Dict[str, t.Any]:
        """Get detailed information about a backend.

        Args:
            name: Name or alias of the backend

        Returns:
            Dictionary with backend information
        """
        backend_class = self[name]
        primary_name = self._resolve_name(name)
        aliases = self._get_aliases_for(primary_name)

        return {
            "name": primary_name,
            "class": backend_class,
            "module": backend_class.__module__,
            "aliases": aliases,
            "doc": backend_class.__doc__ or "No documentation available",
        }

    def list_backend_info(self) -> t.List[t.Dict[str, t.Any]]:
        """List detailed information about all backends.

        Returns:
            List of dictionaries with backend information
        """
        if not self._discovered:
            self.discover_backends()

        return [self.get_backend_info(name) for name in self.keys()]

    def clear(self) -> None:
        """Clear all registered backends. Mainly for testing."""
        self._backends.clear()
        self._aliases.clear()
        self._discovered = False

    def create_backend(self, backend_type: str, **kwargs) -> BaseBackend:
        """Create a backend instance.

        Args:
            backend_type: The type of backend to create
            **kwargs: Arguments to pass to the backend constructor

        Returns:
            BaseBackend: An instance of the requested backend
        """
        backend_class = self[backend_type]
        return backend_class(**kwargs)

    def __getitem__(self, name: str) -> t.Type[BaseBackend]:
        """Get a backend class by name (dict-like access)."""
        if not self._discovered:
            self.discover_backends()
        resolved_name = self._resolve_name(name)

        if resolved_name not in self._backends:
            raise KeyError(
                f"Backend '{name}' not found. Available backends: {self._get_available_names()}"
            )

        return self._backends[resolved_name]

    def __setitem__(self, name: str, backend_class: t.Type[BaseBackend]) -> None:
        """Register a backend class (dict-like assignment)."""
        self._validate_name(name)
        self._validate_backend_class(backend_class)

        self._backends[name] = backend_class
        logger.debug(f"Registered backend: {name} -> {backend_class}")

    def __delitem__(self, name: str) -> None:
        """Unregister a backend (dict-like deletion)."""
        # Check if it's an alias first
        if name in self._aliases:
            del self._aliases[name]
            logger.debug(f"Removed alias: {name}")
            return

        if name not in self._backends:
            raise KeyError(f"Backend '{name}' not found")

        # Remove the backend
        del self._backends[name]
        logger.debug(f"Unregistered backend: {name}")

        # Remove any aliases pointing to this backend
        for alias in self._get_aliases_for(name):
            del self._aliases[alias]
            logger.debug(f"Removed alias: {alias}")

    def __contains__(self, name: str) -> bool:
        """Check if a backend exists (dict-like 'in' operator)."""
        if not self._discovered:
            self.discover_backends()
        return name in self._backends or name in self._aliases

    def __iter__(self) -> t.Iterator[str]:
        """Iterate over backend names (dict-like iteration)."""
        if not self._discovered:
            self.discover_backends()
        return iter(self._backends.keys())

    def __len__(self) -> int:
        """Return number of registered backends (dict-like len())."""
        if not self._discovered:
            self.discover_backends()
        return len(self._backends)

    def keys(self) -> t.KeysView[str]:
        """Return view of backend names."""
        if not self._discovered:
            self.discover_backends()
        return self._backends.keys()

    def values(self) -> t.ValuesView[t.Type[BaseBackend]]:
        """Return view of backend classes."""
        if not self._discovered:
            self.discover_backends()
        return self._backends.values()

    def items(self) -> t.ItemsView[str, t.Type[BaseBackend]]:
        """Return view of (name, backend_class) pairs."""
        if not self._discovered:
            self.discover_backends()
        return self._backends.items()

    def __repr__(self) -> str:
        items = {name: backend_class for name, backend_class in self.items()}
        return repr(items)

    __str__ = __repr__


# Global registry instance
BACKEND_REGISTRY = BackendRegistry()


def get_registry() -> BackendRegistry:
    """Get the global backend registry instance."""
    return BACKEND_REGISTRY


def register_backend(
    name: str,
    backend_class: t.Type[BaseBackend],
    aliases: t.Optional[t.List[str]] = None,
) -> None:
    """Register a backend with the global registry.

    Args:
        name: Primary name for the backend
        backend_class: The backend class to register
        aliases: Optional list of alternative names for the backend
    """
    BACKEND_REGISTRY[name] = backend_class
    if aliases:
        BACKEND_REGISTRY.register_aliases(name, aliases)


def print_available_backends() -> None:
    """Print a formatted list of available backends."""
    backends = BACKEND_REGISTRY.list_backend_info()

    if not backends:
        print("No backends available.")
        return

    print("Available backends:")
    print("-" * 50)

    for backend in backends:
        print(f"Name: {backend['name']}")
        if backend["aliases"]:
            print(f"Aliases: {', '.join(backend['aliases'])}")
        print(f"Module: {backend['module']}")
        print(f"Description: {backend['doc']}")
        print("-" * 50)

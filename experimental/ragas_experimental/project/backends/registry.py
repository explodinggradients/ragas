"""Backend registry for managing and discovering project backends."""

import logging
import typing as t
from importlib import metadata

from .base import ProjectBackend

logger = logging.getLogger(__name__)


class BackendRegistry:
    """Registry for managing project backends with plugin support."""

    _instance = None
    _backends: t.Dict[str, t.Type[ProjectBackend]] = {}
    _aliases: t.Dict[str, str] = {}
    _discovered = False

    def __new__(cls):
        """Singleton pattern to ensure single registry instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def instance(cls) -> "BackendRegistry":
        """Get the singleton registry instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register_backend(
        self,
        name: str,
        backend_class: t.Type[ProjectBackend],
        aliases: t.Optional[t.List[str]] = None,
        overwrite: bool = False,
    ) -> None:
        """Register a backend class with the registry.

        Args:
            name: Primary name for the backend
            backend_class: The backend class to register
            aliases: Optional list of alternative names for the backend
            overwrite: Whether to overwrite existing backends with the same name

        Raises:
            TypeError: If backend_class doesn't inherit from ProjectBackend
            ValueError: If backend name already exists and overwrite=False
        """
        if not name or not isinstance(name, str):
            raise ValueError("Backend name must be a non-empty string")

        if not issubclass(backend_class, ProjectBackend):
            raise TypeError(
                f"Backend class {backend_class} must inherit from ProjectBackend"
            )

        # Check for existing registration
        if name in self._backends and not overwrite:
            raise ValueError(
                f"Backend '{name}' is already registered. Use overwrite=True to replace."
            )

        self._backends[name] = backend_class
        logger.debug(f"Registered backend: {name} -> {backend_class}")

        # Register aliases
        if aliases:
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

    def get_backend(self, name: str) -> t.Type[ProjectBackend]:
        """Get a backend class by name.

        Args:
            name: Name or alias of the backend

        Returns:
            The backend class

        Raises:
            ValueError: If backend is not found
        """
        # Ensure backends are discovered
        if not self._discovered:
            self.discover_backends()

        # Check if it's an alias first
        if name in self._aliases:
            name = self._aliases[name]

        if name not in self._backends:
            available = list(self._backends.keys()) + list(self._aliases.keys())
            raise ValueError(
                f"Backend '{name}' not found. Available backends: {available}"
            )

        return self._backends[name]

    def list_available_backends(self) -> t.List[str]:
        """List all available backend names.

        Returns:
            List of backend names (primary names only, not aliases)
        """
        if not self._discovered:
            self.discover_backends()

        return list(self._backends.keys())

    def list_all_names(self) -> t.Dict[str, t.List[str]]:
        """List all backend names including aliases.

        Returns:
            Dictionary mapping primary names to lists of all names (including aliases)
        """
        if not self._discovered:
            self.discover_backends()

        result = {}
        for primary_name in self._backends.keys():
            aliases = [
                alias
                for alias, target in self._aliases.items()
                if target == primary_name
            ]
            result[primary_name] = [primary_name] + aliases

        return result

    def discover_backends(self) -> t.Dict[str, t.Type[ProjectBackend]]:
        """Discover and register backends from entry points and manual registration.

        Returns:
            Dictionary of discovered backends
        """
        if self._discovered:
            return self._backends.copy()

        logger.debug("Discovering backends...")

        # First register built-in backends manually (for now)
        self._register_builtin_backends()

        # Then discover from entry points
        self._discover_from_entry_points()

        self._discovered = True
        logger.info(
            f"Backend discovery complete. Found {len(self._backends)} backends."
        )

        return self._backends.copy()

    def _register_builtin_backends(self) -> None:
        """Register the built-in backends."""
        try:
            from .local_csv import LocalCSVProjectBackend

            self.register_backend("local/csv", LocalCSVProjectBackend)

            from .platform import PlatformProjectBackend

            self.register_backend("ragas/app", PlatformProjectBackend)

        except ImportError as e:
            logger.warning(f"Failed to import built-in backend: {e}")

    def _discover_from_entry_points(self) -> None:
        """Discover backends from setuptools entry points."""
        try:
            # Look for entry points in the 'ragas.backends' group
            entry_points = metadata.entry_points().select(group="ragas.backends")

            for entry_point in entry_points:
                try:
                    backend_class = entry_point.load()
                    self.register_backend(entry_point.name, backend_class)
                    logger.info(
                        f"Discovered backend from entry point: {entry_point.name}"
                    )

                except Exception as e:
                    logger.warning(f"Failed to load backend '{entry_point.name}': {e}")

        except Exception as e:
            logger.debug(
                f"Entry point discovery failed (this is normal if no plugins installed): {e}"
            )

    def get_backend_info(self, name: str) -> t.Dict[str, t.Any]:
        """Get detailed information about a backend.

        Args:
            name: Name or alias of the backend

        Returns:
            Dictionary with backend information
        """
        backend_class = self.get_backend(name)

        # Resolve to primary name if it's an alias
        primary_name = name
        if name in self._aliases:
            primary_name = self._aliases[name]

        # Get all aliases for this backend
        aliases = [
            alias for alias, target in self._aliases.items() if target == primary_name
        ]

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

        return [self.get_backend_info(name) for name in self._backends.keys()]

    def clear(self) -> None:
        """Clear all registered backends. Mainly for testing."""
        self._backends.clear()
        self._aliases.clear()
        self._discovered = False

    def create_backend(self, backend_type: str, **kwargs) -> ProjectBackend:
        """Create a backend instance.

        Args:
            backend_type: The type of backend to create
            **kwargs: Arguments specific to the backend

        Returns:
            ProjectBackend: An instance of the requested backend
        """
        backend_class = self.get_backend(backend_type)
        return backend_class(**kwargs)


# Global registry instance
_registry = BackendRegistry.instance()


def get_registry() -> BackendRegistry:
    """Get the global backend registry instance."""
    return _registry


def register_backend(
    name: str,
    backend_class: t.Type[ProjectBackend],
    aliases: t.Optional[t.List[str]] = None,
) -> None:
    """Register a backend with the global registry.

    Args:
        name: Primary name for the backend
        backend_class: The backend class to register
        aliases: Optional list of alternative names for the backend
    """
    _registry.register_backend(name, backend_class, aliases)


def list_backends() -> t.List[str]:
    """List all available backend names."""
    return _registry.list_available_backends()


def get_backend_info(name: str) -> t.Dict[str, t.Any]:
    """Get detailed information about a specific backend."""
    return _registry.get_backend_info(name)


def list_backend_info() -> t.List[t.Dict[str, t.Any]]:
    """List detailed information about all available backends."""
    return _registry.list_backend_info()


def print_available_backends() -> None:
    """Print a formatted list of available backends."""
    backends = _registry.list_backend_info()

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


def create_project_backend(backend_type: str, **kwargs) -> ProjectBackend:
    """Create a project backend instance.

    Args:
        backend_type: The type of backend to create
        **kwargs: Arguments specific to the backend

    Returns:
        ProjectBackend: An instance of the requested backend
    """
    return _registry.create_backend(backend_type, **kwargs)

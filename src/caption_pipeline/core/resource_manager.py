"""
ResourceManager: Abstract base class for managing external resources.

Provides a consistent interface for:
- Starting/initializing resources (servers, models, services)
- Stopping/cleaning up resources
- Health/ready checks
- Context manager support

This pattern makes it trivial to add new resource managers for any external
service (e.g., llama-server, Ollama, custom APIs, database connections).
"""

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar, Self

T = TypeVar('T')  # Config type


class ResourceManager(ABC, Generic[T]):
    """
    Abstract base class for managing external resources.

    Any external service that needs lifecycle management (start/stop)
    should implement this interface.

    Example:
        class MyAPIManager(ResourceManager[MyConfig]):
            def start(self) -> bool: ...
            def stop(self) -> bool: ...
            def is_ready(self) -> bool: ...
    """

    def __init__(self, config: T) -> None:
        """
        Initialize the resource manager.

        Args:
            config: Configuration for the resource
        """
        self.config: T = config
        self._started_by_us: bool = False
        self._ready: bool = False

    @abstractmethod
    def start(self) -> bool:
        """
        Start or initialize the resource.

        This should:
        - Establish connections
        - Start subprocesses
        - Load models into memory
        - Wait for readiness

        Returns:
            True if started successfully, False otherwise
        """
        pass

    @abstractmethod
    def stop(self) -> bool:
        """
        Stop or cleanup the resource.

        This should:
        - Terminate subprocesses
        - Unload models from memory
        - Close connections
        - Clean up temporary files

        Returns:
            True if stopped successfully, False otherwise
        """
        pass

    @abstractmethod
    def is_ready(self) -> bool:
        """
        Check if the resource is ready to accept requests.

        Returns:
            True if ready, False otherwise
        """
        pass

    @property
    def started_by_us(self) -> bool:
        """Whether this manager started the resource."""
        return self._started_by_us

    @property
    def is_active(self) -> bool:
        """Whether the resource is currently active."""
        return self._ready

    def __enter__(self) -> Self:
        """Enter context manager - starts the resource."""
        if not self.start():
            raise RuntimeError(f"Failed to start {self.__class__.__name__}")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object | None,
    ) -> None:
        """Exit context manager - stops the resource if we started it."""
        if self._started_by_us:
            self.stop()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"ready={self._ready}, "
            f"started_by_us={self._started_by_us})"
        )

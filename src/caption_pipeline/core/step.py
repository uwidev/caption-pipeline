"""
PipelineStep: The interface that all processing steps must implement.
"""

from abc import ABC, abstractmethod
from typing import Any

from .context import ImageContext


class PipelineStep(ABC):
    """
    Base class for all pipeline operations.
    """

    # Help metadata (set by @step_help decorator)
    _help_meta: Any = None

    @abstractmethod
    def name(self) -> str:
        """Return the step's unique identifier."""
        pass

    @abstractmethod
    def process(self, context: ImageContext) -> ImageContext | None:
        """
        Process a single image context.

        Args:
            context: The current image context

        Returns:
            Modified context, or None if image should be filtered out
        """
        pass

    def process_batch(self, contexts: list[ImageContext]) -> list[ImageContext]:
        """
        Process multiple contexts in batch.

        Default implementation calls process() sequentially.
        Override for batch-optimized processing (e.g., server management).

        Args:
            contexts: List of contexts to process

        Returns:
            List of processed contexts (filtered out contexts removed)
        """
        results: list[ImageContext] = []
        for context in contexts:
            result = self.process(context)
            if result is not None:
                results.append(result)
        return results

    @abstractmethod
    def validate(self, context: ImageContext) -> bool:
        """
        Validate if this step should run on this context.

        Returns:
            True if the step should run, False to skip
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name()}')"

"""
Pipeline: Orchestrates the sequential execution of steps.
"""

import traceback
from typing import Callable, Literal, Self

from caption_pipeline.utils.logging_utils import log

from .context import ImageContext
from .step import PipelineStep


class Pipeline:
    """
    Orchestrates sequential processing of steps.

    Processing flow:
        1. Step 1 on ALL images
        2. Step 2 on ALL images
        3. Step 3 on ALL images
        ...

    Steps can still override process_batch() for resource management
    (e.g., keeping a server running across images).
    """

    def __init__(self, error_handling: Literal["stop", "skip"] = "stop") -> None:
        """
        Initialize the pipeline.

        Args:
            error_handling: 'stop' to halt on error, 'skip' to continue
        """
        self.steps: list[PipelineStep] = []
        self.error_handling: Literal["stop", "skip"] = error_handling

        # Callbacks
        self._on_step_start: Callable[[str, int], None] | None = None
        self._on_step_complete: Callable[[str, int], None] | None = None
        self._on_error: Callable[[str, ImageContext | None, Exception], None] | None = None

    def add_step(self, step: PipelineStep) -> Self:
        """Add a processing step to the pipeline."""
        self.steps.append(step)
        return self

    def on_step_start(self, callback: Callable[[str, int], None]) -> Self:
        """Register a callback for step start events."""
        self._on_step_start = callback
        return self

    def on_step_complete(self, callback: Callable[[str, int], None]) -> Self:
        """Register a callback for step complete events."""
        self._on_step_complete = callback
        return self

    def on_error(self, callback: Callable[[str, ImageContext | None, Exception], None]) -> Self:
        """Register a callback for error events."""
        self._on_error = callback
        return self

    def run(self, contexts: list[ImageContext]) -> list[ImageContext]:
        """
        Execute the pipeline step-by-step on all contexts.

        Flow: Step1 on all images -> Step2 on all images -> ...

        Args:
            contexts: List of image contexts to process

        Returns:
            List of processed contexts (filtered if steps returned None)
        """
        current_contexts = contexts.copy()
        
        for step_idx, step in enumerate(self.steps):
            if not current_contexts:
                log.warning(f"No contexts remaining before step {step.name()}")
                break
            
            log.info(
                f"Running step {step_idx+1}/{len(self.steps)}: {step.name()} "
                f"on {len(current_contexts)} images"
            )
            
            self._notify_step_start(step.name(), len(current_contexts))
            
            try:
                # Process in batch (step handles batching internally)
                current_contexts = step.process_batch(current_contexts)
                
                # Add step to history for each context
                for context in current_contexts:
                    context.add_history(step.name())
                
                self._notify_step_complete(step.name(), len(current_contexts))
                
            except Exception as e:
                tb = traceback.format_exc()
                log.error(f"Step {step.name()} failed:\n{tb}")
                
                ctx = current_contexts[0] if current_contexts else None
                self._notify_error(step.name(), ctx, e)
                
                if self.error_handling == "stop":
                    raise
                # On 'skip', continue to next step with current contexts
        
        return current_contexts

    def _notify_step_start(self, step_name: str, count: int) -> None:
        """Notify step start."""
        if self._on_step_start:
            self._on_step_start(step_name, count)

    def _notify_step_complete(self, step_name: str, count: int) -> None:
        """Notify step complete."""
        if self._on_step_complete:
            self._on_step_complete(step_name, count)

    def _notify_error(self, step_name: str, context: ImageContext | None, error: Exception) -> None:
        """Notify error."""
        if self._on_error:
            self._on_error(step_name, context, error)

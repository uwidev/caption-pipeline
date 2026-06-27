"""
Core components for the caption pipeline.

Includes:
- ImageContext: Data container
- Pipeline: Orchestrator
- PipelineStep: Step abstraction
- ResourceManager: Resource lifecycle management
- Help decorators and utilities
"""

from caption_pipeline.core.context import ImageContext
from caption_pipeline.core.pipeline import Pipeline
from caption_pipeline.core.step import PipelineStep
from caption_pipeline.core.resource_manager import ResourceManager
from caption_pipeline.core.help import step_help, get_step_help, format_step_help

__all__ = [
    "ImageContext",
    "Pipeline",
    "PipelineStep",
    "ResourceManager",
    "step_help",
    "get_step_help",
    "format_step_help",
]

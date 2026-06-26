"""
Caption Pipeline - Modular image captioning pipeline for diffusion model training.

This package provides a flexible pipeline for processing images and generating
captions for diffusion model training. It supports:
- AI-based tag generation (WD14, PixAI)
- CLIP token resolution
- Tag manipulation and formatting
- Extensible pipeline architecture

Example:
    >>> from caption_pipeline import Pipeline, ImageContext
    >>> from caption_pipeline.steps import TagGenerationStep, FormatJoinStep
    >>>
    >>> pipeline = Pipeline()
    >>> pipeline.add_step(TagGenerationStep(threshold=0.35))
    >>> pipeline.add_step(FormatJoinStep())
    >>>
    >>> context = ImageContext(image_path=Path("image.png"), source_path=Path("image.png"))
    >>> result = pipeline.run([context])
"""

from caption_pipeline.core import ImageContext, Pipeline, PipelineStep
from caption_pipeline.steps import (
    FormatJoinStep,
    TagGenerationStep,
    TagManipulateStep,
    TagNaturalLanguageFilterStep,
    TagNaturalLanguageStep,
    TagResolveStep,
)

__version__ = "0.1.0"
__all__ = [
    # Core
    "ImageContext",
    "Pipeline",
    "PipelineStep",
    # Steps
    "TagGenerationStep",
    "TagResolveStep",
    "TagManipulateStep",
    "TagNaturalLanguageStep",
    "TagNaturalLanguageFilterStep",
    "FormatJoinStep",
]

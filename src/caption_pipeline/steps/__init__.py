"""
Pipeline steps for image captioning operations.

This module provides ready-to-use steps for common captioning operations:
- TagGenerationStep: Generate tags using AI models (WD14, PixAI)
- TagResolveStep: Resolve tags within CLIP token limits
- TagManipulateStep: Add, remove, or reorder tags
- TagNaturalLanguageStep: Generate NL captions using ToriiGate
- TagNaturalLanguageFilterStep: Filter NL captions through Ollama
- FormatJoinStep: Format and save final captions
- CharacterValidationStep: Validate character tags in grounding hints
- DebugStep: Debug system state

Each step is a self-contained operation that can be composed into a pipeline.
Steps are designed to be stateless and idempotent when possible.
"""

from caption_pipeline.steps.format_join import FormatJoinStep
from caption_pipeline.steps.tag_generate import TagGenerationStep
from caption_pipeline.steps.tag_manipulate import TagManipulateStep
from caption_pipeline.steps.tag_natural_language import TagNaturalLanguageStep
from caption_pipeline.steps.tag_natural_language_filter import TagNaturalLanguageFilterStep
from caption_pipeline.steps.tag_resolve import TagResolveStep
from caption_pipeline.steps.validate_characters import CharacterValidationStep
from caption_pipeline.steps.debug import DebugStep

__all__ = [
    "TagGenerationStep",
    "TagResolveStep",
    "TagManipulateStep",
    "TagNaturalLanguageStep",
    "TagNaturalLanguageFilterStep",
    "FormatJoinStep",
    "CharacterValidationStep",
    "DebugStep",
]

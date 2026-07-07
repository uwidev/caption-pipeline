"""
Prompt templates for various pipeline steps.

This module contains prompt templates used by different pipeline steps:
- TORIIGATE_PROMPTS: Prompt templates for ToriiGate NL generation
- NL_FILTER_SYSTEM_PROMPT: System prompt for filtering artstyle references
- FORMAT_NEWLINE_SYSTEM_PROMPT: System prompt for tag categorization

These prompts are externalized to keep the main step implementations lean
and to make it easier to modify or extend prompts without touching the
core logic.
"""

from caption_pipeline.prompts.toriigate import TORIIGATE_PROMPTS
from caption_pipeline.prompts.natural_language import NL_NO_STYLE_SYSTEM_PROMPT
from caption_pipeline.prompts.categorize_tag import CATEGORIZE_TAG_SYSTEM_PROMPT

__all__ = [
    "TORIIGATE_PROMPTS",
    "NL_NO_STYLE_SYSTEM_PROMPT",
    "CATEGORIZE_TAG_SYSTEM_PROMPT",
]

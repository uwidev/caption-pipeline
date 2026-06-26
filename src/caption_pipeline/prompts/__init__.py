"""
Prompt templates for various pipeline steps.

This module contains prompt templates used by different pipeline steps:
- TORIIGATE_PROMPTS: Prompt templates for ToriiGate NL generation

These prompts are externalized to keep the main step implementations lean
and to make it easier to modify or extend prompts without touching the
core logic.
"""

from caption_pipeline.prompts.toriigate import TORIIGATE_PROMPTS
from caption_pipeline.prompts.filter import NL_FILTER_SYSTEM_PROMPT

__all__ = [
    "TORIIGATE_PROMPTS",
    "NL_FILTER_SYSTEM_PROMPT"
]

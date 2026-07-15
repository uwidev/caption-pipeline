"""
Utility functions and helpers for the caption pipeline.
"""

from caption_pipeline.utils.image_utils import (
    SUPPORTED_IMAGE_MIMES,
    find_images_in_directory,
    is_image_file,
)
from caption_pipeline.utils.llama_server import LlamaServer, LlamaServerConfig
from caption_pipeline.utils.logging_utils import (
    configure_logging,
    log,
    log_list_truncated,
    log_scored_list_truncated,
    log_truncated,
    section,
)
from caption_pipeline.utils.ollama_manager import OllamaConfig, OllamaManager
from caption_pipeline.utils.tag_db import (
    get_all_characters,
    get_character_description,
    get_character_info,
    get_character_names,
    get_character_popular_tags,
    get_display_name,
    get_parent_tag,
    is_alias,
    is_skin,
    load_character_data,
    load_character_tags_only,
    load_general_tags_only,
    load_tag_databases,
    query_character,
    query_character_field,
    search_characters,
    add_custom_character,
)
from caption_pipeline.utils.tag_patterns import PRE_CATEGORIZE_PATTERNS
from caption_pipeline.utils.tag_utils import sort_key_by_object, sort_tags_by_object
from caption_pipeline.utils.tokenizer import get_tokenizer

__all__ = [
    # image_utils
    "SUPPORTED_IMAGE_MIMES",
    "is_image_file",
    "find_images_in_directory",
    # tag_db
    "load_tag_databases",
    "load_character_tags_only",
    "load_general_tags_only",
    "load_character_data",
    "query_character",
    "query_character_field",
    "get_character_popular_tags",
    "get_character_description",
    "get_display_name",
    "get_parent_tag",
    "is_alias",
    "is_skin",
    "get_character_info",
    "search_characters",
    "get_all_characters",
    "get_character_names",
    # tokenizer
    "get_tokenizer",
    # Resource managers
    "OllamaManager",
    "OllamaConfig",
    "LlamaServer",
    "LlamaServerConfig",
    # Logging
    "log",
    "configure_logging",
    "section",
    "log_truncated",
    "log_list_truncated",
    "log_scored_list_truncated",
    # Tag patterns
    "PRE_CATEGORIZE_PATTERNS",
    # Tag utils
    "sort_key_by_object",
    "sort_tags_by_object",
    "add_custom_character"
]

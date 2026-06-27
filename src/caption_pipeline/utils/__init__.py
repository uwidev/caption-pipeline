"""
Utility functions and helpers for the caption pipeline.
"""

from caption_pipeline.utils.tag_db import (
    # Database loading
    load_tag_databases,
    load_character_tags_only,
    load_general_tags_only,
    load_character_data,
    # Character queries
    query_character,
    query_character_field,
    get_character_popular_tags,
    get_character_description,
    get_display_name,
    get_parent_tag,
    is_alias,
    is_skin,
    get_character_info,
    search_characters,
    get_all_characters,
    get_character_names,
)
from caption_pipeline.utils.tokenizer import get_tokenizer
from caption_pipeline.utils.ollama_manager import (
    OllamaManager,
    OllamaConfig,
)
from caption_pipeline.utils.llama_server import (
    LlamaServer,
    LlamaServerConfig
)
from caption_pipeline.utils.model_manager import (
    ModelManager
)
from caption_pipeline.utils.logging_utils import IndentedLogger, log

__all__ = [
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
    "ModelManager",
    #Logging
    "IndentedLogger",
    "log",
]

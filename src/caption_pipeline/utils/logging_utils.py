"""
Logging utilities for the caption pipeline.

Provides:
- configure_logging: Set up loguru with indentation support
- section: Context manager for indented log sections
- log_truncated: Log truncated content with continuation
- log_list_truncated: Log lists with truncation
- log_scored_list_truncated: Log scored lists with truncation
- log: Direct access to loguru logger

Logging is a redirection to this module to allow for context-
aware indentation. All logging should use `log` to ensure
proper indentation.

Due to the redirection, it adds a frame to the stack frame.
This results in the logging context to be from this module
rather than the original caller. To fix that, we need to
go up a frame by using `depth`. If it's a redirection on top
of a redirection (see `log_scored_list_truncated()`), it adds
two frames, so `depth=2`.
"""

import sys
from contextlib import contextmanager
from typing import Any

from loguru import logger

# Global indentation state
_indent_level: list[int] = [0]
_INDENT_STR: str = "  "


def configure_logging(debug: bool = False) -> None:
    """
    Configure loguru with indentation support and colored output.

    Args:
        debug: Enable debug-level logging
    """
    # Remove default handlers
    logger.remove()

    def add_indentation(record: dict[str, Any]) -> bool:
        """Add indentation to every log record and clean up module names."""
        record["extra"]["indent"] = _INDENT_STR * _indent_level[0]
        
        # Clean up module name to just the last part
        # The 'name' field contains the full module path (e.g., 'caption_pipeline.cli')
        if record.get("name"):
            # Split by '.' and get the last part
            record["name"] = record["name"].split(".")[-1]
        return True

    if debug:
        level = "DEBUG"
        format_str = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan> - "
            "{extra[indent]}"
            "<level>{message}</level>"
        )
    else:
        level = "INFO"
        format_str = (
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan> - "
            "{extra[indent]}"
            "<level>{message}</level>"
        )

    # Add stdout sink with our formatter
    logger.add(
        sys.stdout,
        level=level,
        format=format_str,
        colorize=True,
        filter=add_indentation,
    )

    # Silence noisy loggers
    import logging

    for logger_name in [
        "httpx",
        "httpcore",
        "huggingface_hub",
        "transformers",
        "filelock",
        "urllib3",
        "PIL",
    ]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


@contextmanager
def section(message: str, level: str = "info"):
    """
    Log a section header and indent all subsequent logs within this context.

    Args:
        message: Section header message
        level: Log level for the header ('info', 'debug', 'warning', etc.)

    Yields:
        None

    Example:
        with section("Processing images"):
            logger.info("Found 10 images")  # This will be indented
            with section("Processing image 1"):
                logger.debug("Loading...")  # Double indented
    """
    # Use depth=2 to skip the @contextmanager wrapper
    getattr(logger.opt(depth=2), level)(message)
    _indent_level[0] += 1
    try:
        yield
    finally:
        _indent_level[0] -= 1


def log_truncated(
    message: str,
    content: str,
    max_len: int = 64,
    level: str = "info",
    continuation_level: str = "debug",
) -> None:
    """
    Log a message with truncated content.

    If content is longer than max_len, logs the preview at the specified level
    and the remainder at continuation_level.

    Args:
        message: The prefix message (e.g., "Wrote", "Output", "NL")
        content: The content to log
        max_len: Maximum characters to show in the first log (default: 64)
        level: Log level for the preview (default: "info")
        continuation_level: Log level for the remainder (default: "debug")
    """
    if level == continuation_level:
        getattr(logger.opt(depth=1), level)(f"{message}: {content}")

    clen = len(content)
    preview = content[:max_len]
    getattr(logger.opt(depth=1), level)(f"{message}: {preview}{'...' if clen > max_len else ''}")

    if clen > max_len:
        remainder = content[max_len:]
        getattr(logger.opt(depth=1), continuation_level)(f"  {remainder}")


def log_list_truncated(
    items: list[str],
    message: str,
    max_items: int = 5,
    level: str = "info",
    continuation_level: str = "debug",
    frame_depth: int = 1,
) -> None:
    """
    Log a numbered list with truncation.

    If items length > max_items, shows first max_items at the specified level
    and the rest at continuation_level.

    Args:
        items: List of items to log
        message: The prefix message (e.g., "Final tags", "Removed tags")
        max_items: Maximum items to show before truncating (default: 5, -1 = show all)
        level: Log level for the header and visible items (default: "info")
        continuation_level: Log level for the continuation (default: "debug")
        frame_depth: How much to go up stack frame for logging context (default: "1")
    """
    total = len(items)
    getattr(logger.opt(depth=frame_depth), level)(f"{message} ({total}):")

    if total <= max_items or max_items == -1 or level == continuation_level:
        for i, item in enumerate(items, 1):
            getattr(logger.opt(depth=frame_depth), level)(f"  {i:>3d}. {item}")
    else:
        for i, item in enumerate(items[:max_items], 1):
            getattr(logger.opt(depth=frame_depth), level)(f"  {i:>3d}. {item}")
        getattr(logger.opt(depth=frame_depth), level)(f"     ...")
        for i, item in enumerate(items[max_items:], max_items + 1):
            getattr(logger.opt(depth=frame_depth), continuation_level)(f"  {i:>3d}. {item}")


def log_scored_list_truncated(
    items: list[tuple[str, float]],
    message: str,
    max_items: int = 5,
    level: str = "info",
    continuation_level: str = "debug",
    frame_depth: int = 2,
) -> None:
    """
    Log a numbered list of scored items with truncation.

    Items are formatted as "tag (score)" and shown as a numbered list.

    Args:
        items: List of (tag, score) tuples
        message: The prefix message (e.g., "Tags above threshold")
        max_items: Maximum items to show before truncating (default: 5)
        level: Log level for the header and visible items (default: "info")
        continuation_level: Log level for the continuation (default: "debug")
        frame_depth: How much to go up stack frame for logging context (default: "1")
    """
    if not items:
        return

    formatted = [f"{tag} ({score:.3f})" for tag, score in items]
    log_list_truncated(
        items=formatted,
        message=message,
        max_items=max_items,
        level=level,
        continuation_level=continuation_level,
        frame_depth=frame_depth,
    )


# Export logger for direct use
log = logger

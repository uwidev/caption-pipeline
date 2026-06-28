"""
Logging utilities for the caption pipeline.

Provides:
- IndentedLogger: Wrapper for loguru with indentation support
- log: Global instance for consistent logging
- log_truncated: Function for logging truncated output with continuation
"""

from contextlib import contextmanager
from loguru import logger
from typing import Any


class IndentedLogger:
    """Wrapper for loguru that manages indentation levels for structured logging."""
    
    def __init__(self, indent_str: str = "  "):
        self._indent_level = 0
        self._indent_str = indent_str
        self._logger = logger
    
    @contextmanager
    def section(self, message: str, level: str = "info"):
        """Log a section header and indent all subsequent logs."""
        self._log(level, message)
        self._indent_level += 1
        try:
            yield
        finally:
            self._indent_level -= 1
    
    def _log(self, level: str, message: str, *args, **kwargs):
        """Internal log method with indentation."""
        indent = self._indent_str * self._indent_level
        # Preserve any extra indentation already in the message
        if message.startswith(self._indent_str):
            indent = ""
        getattr(self._logger, level)(f"{indent}{message}", *args, **kwargs)
    
    def info(self, message: str, *args, **kwargs):
        self._log("info", message, *args, **kwargs)
    
    def debug(self, message: str, *args, **kwargs):
        self._log("debug", message, *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs):
        self._log("warning", message, *args, **kwargs)
    
    def error(self, message: str, *args, **kwargs):
        self._log("error", message, *args, **kwargs)
    
    def success(self, message: str, *args, **kwargs):
        self._log("success", message, *args, **kwargs)


# Global instance
log = IndentedLogger()


def log_truncated(
    message: str,
    content: str,
    max_len: int = 64,
    level: str = "info",
    continuation_level: str = "debug",
    indent: int = 0,
) -> None:
    """
    Log a message with truncated content, showing continuation in a separate log.
    
    If content is longer than max_len and level != continuation_level, logs:
        INFO: message: first 64 chars...
        DEBUG: continuation: rest of content
    
    If level == continuation_level, logs the entire content in one go.
    If content is shorter, logs only the INFO line.
    
    Args:
        message: The prefix message (e.g., "Wrote", "Output", "NL")
        content: The content to log
        max_len: Maximum characters to show in the first log (default: 64)
        level: Log level for the first line (default: "info")
        continuation_level: Log level for the continuation (default: "debug")
        indent: Indentation level (default: 0)
    """
    if not content:
        return
    
    indent_str = "  " * indent
    full_message = f"{indent_str}{message}: "
    
    # If level and continuation_level are the same, just log the entire thing
    # You honestly should just be using 'log' normally, but I suppose this
    # allows you to prefix the message somewhat.
    if level == continuation_level or len(content) <= max_len:
        getattr(log, level)(f"{full_message}{content}")
    else:
        # Truncate and show continuation
        preview = content[:max_len]
        remainder = content[max_len:]
        getattr(log, level)(f"{full_message}{preview}...")
        getattr(log, continuation_level)(f"{indent_str}  {remainder}")


def log_list_truncated(
    items: list[str],
    message: str,
    max_items: int = 5,
    level: str = "info",
    continuation_level: str = "debug",
    indent_str: str = "  ",
) -> None:
    """
    Log a numbered list with truncation, showing continuation at a lower level.

    If items length > max_items and level != continuation_level, logs:
        INFO: message (12):
        INFO:   1. item1
        INFO:   2. item2
        INFO:   3. item3
        INFO:   4. item4
        INFO:   5. item5
        INFO:   ...
        DEBUG:   6. item6
        DEBUG:   7. item7
        ...

    If level == continuation_level, logs all items at the same level without ellipsis.
    If items length <= max_items, logs all items at the same level.

    Args:
        items: List of items to log
        message: The prefix message (e.g., "Final tags", "Removed tags")
        max_items: Maximum items to show before truncating (default: 5, -1 show all on level)
        level: Log level for the first line and visible items (default: "info")
        continuation_level: Log level for the continuation (default: "debug")
        indent: Indentation level (default: 0)
    """
    if not items:
        return

    total = len(items)

    # Log header with count
    getattr(log, level)(f"{message} ({total}):")

    # If level == continuation_level, show all items at the same level
    if level == continuation_level or total <= max_items or max_items == -1:
        for i, item in enumerate(items, 1):
            getattr(log, level)(f"{indent_str}  {i:>3d}. {item}")
    else:
        # Show first max_items at the specified level
        for i, item in enumerate(items[:max_items], 1):
            getattr(log, level)(f"{indent_str}  {i:>3d}. {item}")

        # Show ellipsis at the specified level
        getattr(log, level)(f"{indent_str}  {' ':>3}  ...")

        # Show remaining items at continuation_level
        for i, item in enumerate(items[max_items:], max_items + 1):
            getattr(log, continuation_level)(f"{indent_str}  {i:>3d}. {item}")


def log_scored_list_truncated(
    items: list[tuple[str, float]],
    message: str,
    max_items: int = 5,
    level: str = "info",
    continuation_level: str = "debug",
    indent_str: str = "  ",
) -> None:
    """
    Log a numbered list of scored items with truncation.

    Items are formatted as "tag (score)" and shown as a numbered list.
    If items length > max_items and level != continuation_level, shows first max_items at INFO and rest at DEBUG.
    If level == continuation_level, shows all items at the same level.

    Args:
        items: List of (tag, score) tuples
        message: The prefix message (e.g., "Tags above threshold")
        max_items: Maximum items to show before truncating (default: 5)
        level: Log level for the first line and visible items (default: "info")
        continuation_level: Log level for the continuation (default: "debug")
        indent: Indentation level (default: 0)
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
        indent_str=indent_str,
    )

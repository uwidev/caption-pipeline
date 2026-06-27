"""
Custom logger with context-aware indentation and visual hierarchy.
"""

from contextlib import contextmanager
from loguru import logger
from typing import Literal, Any
import threading


class IndentedLogger:
    """
    Wrapper for loguru that manages indentation levels for structured logging.
    
    Features:
    - Visual indentation using vertical bars (│) instead of spaces
    - Proper caller attribution (shows the actual calling module, not logging_utils)
    - Configurable indent style (bars, spaces, or dots)
    - Optional thread/process IDs for debugging
    """
    
    def __init__(
        self,
        indent_str: str = "  ",
        indent_style: Literal["bars", "spaces", "dots"] = "spaces",
        show_caller: bool = True,
        show_thread: bool = False,
    ) -> None:
        """
        Initialize the indented logger.
        
        Args:
            indent_str: Base string for each indent level (default: "  ")
            indent_style: Visual style for indentation
                - "bars": Uses vertical bars (│) for clear hierarchy
                - "spaces": Uses spaces (clean but harder to distinguish)
                - "dots": Uses dots (subtle visual cue)
            show_caller: Whether to show the calling module/function
            show_thread: Whether to show thread ID (for debugging concurrency)
        """
        self._indent_level: int = 0
        self._indent_str: str = indent_str
        self._indent_style: Literal["bars", "spaces", "dots"] = indent_style
        self._show_caller: bool = show_caller
        self._show_thread: bool = show_thread
        self._logger = logger
        
        # Pre-compute indent characters for speed
        self._indent_chars: dict[str, str] = {
            "bars": "│",
            "spaces": " ",
            "dots": "·",
        }
    
    @contextmanager
    def section(self, message: str, level: str = "info"):
        """
        Log a section header and indent all subsequent logs.
        
        Example:
            with log.section("Processing batch"):
                log.info("Step 1 complete")
                log.info("Step 2 complete")
            # Indentation returns to previous level
        """
        self._log(level, f"─── {message} ───")
        self._indent_level += 1
        try:
            yield
        finally:
            self._indent_level -= 1
    
    @contextmanager
    def section_debug(self, message: str):
        """Section header at DEBUG level."""
        with self.section(message, "debug"):
            yield
    
    def _get_indent(self) -> str:
        """Build the indentation string for the current level."""
        if self._indent_level == 0:
            return ""
        
        char = self._indent_chars.get(self._indent_style, " ")
        
        # For bars, create a tree-like structure
        if self._indent_style == "bars":
            # Level 1: "│ "
            # Level 2: "│ │ "
            # Level 3: "│ │ │ "
            return ("│ " * (self._indent_level - 1)) + "├─ "
        else:
            # Simple repetition for other styles
            return char * self._indent_level + " "
    
    def _log(self, level: str, message: str, *args: Any, **kwargs: Any) -> None:
        """
        Internal log method with indentation and proper caller attribution.
        
        depth=2 skips:
            1. self._log() itself
            2. The wrapper method (info/debug/warning/etc.)
        This shows the ACTUAL caller (the module that called log.info()).
        """
        indent = self._get_indent()
        
        # Add thread/process info if requested
        if self._show_thread:
            thread_info = f"[T{threading.current_thread().ident % 1000}] "
            message = f"{thread_info}{message}"
        
        # Call loguru with depth=2 for correct caller attribution
        getattr(self._logger.opt(depth=2), level)(f"{indent}{message}", *args, **kwargs)
    
    # ===== Public API =====
    
    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._log("info", message, *args, **kwargs)
    
    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._log("debug", message, *args, **kwargs)
    
    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._log("warning", message, *args, **kwargs)
    
    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._log("error", message, *args, **kwargs)
    
    def success(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._log("success", message, *args, **kwargs)
    
    def trace(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._log("trace", message, *args, **kwargs)
    
    # ===== Advanced Methods =====
    
    def set_level(self, level: str) -> None:
        """Change the logging level dynamically."""
        self._logger.level(level)
    
    def indent_manual(self, level: int = 1) -> None:
        """Manually adjust indentation level."""
        self._indent_level = max(0, self._indent_level + level)
    
    def reset_indent(self) -> None:
        """Reset indentation to zero."""
        self._indent_level = 0
    
    @contextmanager
    def section_depth(self, message: str, depth: int = 1):
        """
        Log a section header at a specific depth level.
        
        Useful for nested sections where you want to control depth manually.
        """
        with self.section(message):
            # Temporarily adjust depth
            self._indent_level += depth - 1
            yield
            self._indent_level -= depth - 1
    
    # ===== Compatibility =====
    
    @property
    def level(self) -> int:
        """Get current indentation level."""
        return self._indent_level
    
    @level.setter
    def level(self, value: int) -> None:
        """Set current indentation level."""
        self._indent_level = max(0, value)
    
    def __repr__(self) -> str:
        return f"IndentedLogger(level={self._indent_level}, style={self._indent_style})"


# ===== Global Instance =====

# Default instance with bars for clear visual hierarchy
log = IndentedLogger(indent_style="bars")

# Alternative instances for different use cases:
# log_debug = IndentedLogger(indent_style="dots", show_thread=True)
# log_compact = IndentedLogger(indent_style="spaces")

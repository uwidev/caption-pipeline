"""
Help metadata decorator for pipeline steps.
"""

from typing import Any, Callable, TypeVar
from functools import wraps

T = TypeVar('T', bound=type)


class StepHelp:
    """Container for step help metadata."""
    
    def __init__(
        self,
        name: str,
        description: str,
        options: list[dict[str, str]] | None = None,
        example: str | None = None,
        long_description: str | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.options = options or []
        self.example = example
        self.long_description = long_description


def step_help(
    name: str,
    description: str,
    options: list[dict[str, str]] | None = None,
    example: str | None = None,
    long_description: str | None = None,
) -> Callable[[T], T]:
    """
    Decorator to attach help metadata to a pipeline step class.
    
    Args:
        name: Step identifier (e.g., "tag:generate")
        description: Short description of what the step does
        options: List of option dicts with 'flag', 'help', 'default' keys
        example: Example usage string
        long_description: Detailed description (multi-line)
    
    Returns:
        Decorated class with _help_meta attribute
    """
    def decorator(cls: T) -> T:
        cls._help_meta = StepHelp(
            name=name,
            description=description,
            options=options,
            example=example,
            long_description=long_description,
        )
        return cls
    
    return decorator


def get_step_help(cls: type) -> StepHelp | None:
    """Get help metadata for a step class."""
    return getattr(cls, '_help_meta', None)


def get_all_step_help(step_classes: list[type]) -> dict[str, StepHelp]:
    """Get help metadata for all step classes."""
    result: dict[str, StepHelp] = {}
    for cls in step_classes:
        meta = get_step_help(cls)
        if meta:
            result[meta.name] = meta
    return result


def format_step_help(meta: StepHelp) -> str:
    """Format a StepHelp object as a string."""
    lines = []
    
    lines.append(f"{meta.name}")
    lines.append("-" * len(meta.name))
    lines.append("")
    lines.append(meta.description)
    lines.append("")
    
    if meta.long_description:
        lines.append(meta.long_description)
        lines.append("")
    
    if meta.options:
        lines.append("Options:")
        lines.append("")
        # Find max flag length for alignment
        max_flag_len = max(len(opt['flag']) for opt in meta.options)
        for opt in meta.options:
            flag = opt['flag'].ljust(max_flag_len)
            default = f" (default: {opt.get('default', 'N/A')})" if 'default' in opt else ""
            lines.append(f"  {flag}  {opt['help']}{default}")
        lines.append("")
    
    if meta.example:
        lines.append("Example:")
        lines.append(f"  {meta.example}")
        lines.append("")
    
    return "\n".join(lines)


def format_all_step_help(step_classes: list[type]) -> str:
    """Format help for all steps."""
    lines = ["STEP REFERENCE", "=" * 80, ""]
    for cls in step_classes:
        meta = get_step_help(cls)
        if meta:
            lines.append(format_step_help(meta))
            lines.append("")
    return "\n".join(lines)

"""
TagManipulateStep: Add, remove, reorder, or move tags.
"""

from typing import Literal

from caption_pipeline.utils.logging_utils import log, section

from caption_pipeline.core.context import ImageContext
from caption_pipeline.core.help import step_help
from caption_pipeline.core.step import PipelineStep


@step_help(
    name="tag:manipulate",
    description="Manipulate tags in a specific section.",
    long_description="""Operations:
  prepend:  Add tags to the beginning of the section
  append:   Add tags to the end (moves existing tags to end)
  replace:  Replace all tags in the section
  remove:   Remove specified tags from the section
  move:     Move specified tags to a new position""",
    options=[
        {
            "flag": "--operation {prepend,append,replace,remove,move}",
            "help": "Operation to perform",
            "default": "prepend",
        },
        {"flag": "--tags TAG,TAG,...", "help": "Tags to manipulate"},
        {
            "flag": "--section INT",
            "help": "Section to modify (0=prepended, 1=main, 2=NL)",
            "default": "1",
        },
        {
            "flag": "--no-remove-duplicates",
            "help": "Allow duplicate tags",
            "default": "remove duplicates",
        },
        {
            "flag": "--target-position INT",
            "help": "For 'move': -1=end, 0=beginning, N=specific index",
            "default": "-1",
        },
    ],
    example="tag:manipulate --operation prepend --tags '@nep (nep 76)' --section 0",
)
class TagManipulateStep(PipelineStep):
    """
    Manipulate tags in a specific section.

    Operations:
    - prepend: Add tags to the beginning
    - append: Add tags to the end (moves existing tags to end)
    - replace: Replace all tags with new tags
    - remove: Remove specified tags
    - move: Move specified tags to a new position (end or specific index)
    """

    def __init__(
        self,
        operation: Literal["prepend", "append", "replace", "remove", "move"],
        tags: list[str],
        target_section: int = 1,  # Renamed from 'section'
        remove_duplicates: bool = True,
        target_position: int = -1,  # -1 = end, 0 = beginning, or specific index
    ):
        """
        Initialize the tag manipulation step.

        Args:
            operation: The operation to perform
            tags: Tags to manipulate
            target_section: Which section to modify
            remove_duplicates: Whether to remove duplicates after operation
            target_position: For 'move' operation, where to move tags to
                            -1 = end, 0 = beginning, >0 = specific index
        """
        self.operation = operation
        self.tags = tags
        self.section = target_section
        self.remove_duplicates = remove_duplicates
        self.target_position = target_position

    def name(self) -> str:
        return f"tag:manipulate:{self.operation}"

    def validate(self, context: ImageContext) -> bool:
        """Always run if tags provided."""
        return bool(self.tags)

    def process(self, context: ImageContext) -> ImageContext | None:
        """Apply tag manipulation."""
        with section(f"Processing: {context.image_path.name}"):
            current_tags = context.get_tags(section=self.section)

            if self.operation == "prepend":
                new_tags = self._prepend_operation(current_tags)
            elif self.operation == "append":
                new_tags = self._append_operation(current_tags)
            elif self.operation == "replace":
                new_tags = self._replace_operation(current_tags)
            elif self.operation == "remove":
                new_tags = self._remove_operation(current_tags)
            elif self.operation == "move":
                new_tags = self._move_operation(current_tags)
            else:
                raise ValueError(f"Unknown operation: {self.operation}")

            # Remove duplicates if requested
            if self.remove_duplicates:
                new_tags = self._remove_duplicates_keep_order(new_tags)

            result = context.copy()
            result.set_tags(new_tags, section=self.section)

            log.debug(f"Applied {self.operation}: {len(current_tags)} -> {len(new_tags)} tags")
            return result

    def _prepend_operation(self, current_tags: list[str]) -> list[str]:
        """Add tags to the beginning."""
        return self.tags + current_tags

    def _append_operation(self, current_tags: list[str]) -> list[str]:
        """
        Add tags to the end.

        If a tag already exists, it will be moved to the end
        (duplicate removal with last occurrence preservation).
        """
        # First, remove any existing tags that we're going to append
        # This ensures they get moved to the end
        filtered = [t for t in current_tags if t not in self.tags]
        return filtered + self.tags

    def _replace_operation(self, current_tags: list[str]) -> list[str]:
        """Replace all tags with new tags."""
        return self.tags

    def _remove_operation(self, current_tags: list[str]) -> list[str]:
        """Remove specified tags."""
        return [t for t in current_tags if t not in self.tags]

    def _move_operation(self, current_tags: list[str]) -> list[str]:
        """
        Move specified tags to a new position.

        - target_position = -1: Move to end (default)
        - target_position = 0: Move to beginning
        - target_position > 0: Move to specific index
        """
        # Remove the tags we're moving
        remaining = [t for t in current_tags if t not in self.tags]

        # If no tags to move, return unchanged
        if not any(t in current_tags for t in self.tags):
            return current_tags

        # Get the tags to move (preserving their order from self.tags)
        # But only those that actually exist in current_tags
        tags_to_move = [t for t in self.tags if t in current_tags]

        # Insert at target position
        if self.target_position == -1:  # End
            return remaining + tags_to_move
        elif self.target_position == 0:  # Beginning
            return tags_to_move + remaining
        else:  # Specific index
            # Clamp to valid range
            pos = min(self.target_position, len(remaining))
            result = remaining[:pos] + tags_to_move + remaining[pos:]
            return result

    def _remove_duplicates_keep_order(self, tags: list[str]) -> list[str]:
        """
        Remove duplicates while preserving order.

        Unlike using set(), this preserves the order of first occurrence.
        """
        seen = set()
        result = []
        for tag in tags:
            if tag not in seen:
                seen.add(tag)
                result.append(tag)
        return result

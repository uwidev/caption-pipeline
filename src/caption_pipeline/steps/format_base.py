"""
BaseFormatStep: Base class for formatting steps.
"""

from pathlib import Path
from typing import Any

from caption_pipeline.core.context import ImageContext
from caption_pipeline.core.step import PipelineStep
from caption_pipeline.utils import log_list_truncated
from caption_pipeline.utils.logging_utils import log


class BaseFormatStep(PipelineStep):
    """
    Base class for formatting steps.

    Provides shared functionality:
    - Tag formatting with delimiter and spacing
    - Output saving
    - Logging a breakdown of tag categories (for section 1)
    """

    SPECIAL_TAGS = {"original", "borrowed_character"}

    def __init__(
        self,
        section: int,
        output_dir: Path | None = None,
        suffix: str = "",
        delimiter: str = ", ",
        use_spaces: bool = True,
    ) -> None:
        """
        Initialize the base format step.

        Args:
            section: Section to format (0, 1, 2)
            output_dir: Output directory for the file
            suffix: Suffix to add to the output filename
            delimiter: Delimiter for tags (tags sections)
            use_spaces: Convert underscores to spaces in tags
        """
        self.section: int = section
        self.output_dir: Path = output_dir or Path("./done/")
        self.suffix: str = suffix
        self.delimiter: str = delimiter
        self.use_spaces: bool = use_spaces

    def _categorize_tags(
        self, tags: list[str], context: ImageContext
    ) -> tuple[list[str], dict[str, Any]]:
        """
        Categorize tags for logging (does NOT reorder).

        Returns:
            Tuple of (tags_list, breakdown)
            - tags_list: the original tags (unchanged)
            - breakdown: dict with 'rating', 'special', 'characters', 'general' lists
        """
        if self.section != 1:
            # For sections 0 and 2, we just return the tags as-is and a simple breakdown
            return tags, {"count": len(tags), "preview": tags[:5]}

        rating = context.rating
        character_tags = context.get_character_tags()

        # Categorize the existing tags
        special_tags = []
        character_found = []
        general_tags = []
        rating_found = None

        for tag in tags:
            if rating and tag == rating:
                rating_found = tag
            elif tag in self.SPECIAL_TAGS:
                special_tags.append(tag)
            elif tag in character_tags:
                character_found.append(tag)
            else:
                general_tags.append(tag)

        breakdown = {
            "rating": rating_found,
            "special": special_tags,
            "characters": character_found,
            "general": general_tags,
        }

        # If we want to display with spaces (for logging), convert if needed
        if self.use_spaces:
            breakdown_display = {
                "rating": rating_found.replace("_", " ") if rating_found else None,
                "special": [t.replace("_", " ") for t in special_tags],
                "characters": [t.replace("_", " ") for t in character_found],
                "general": [t.replace("_", " ") for t in general_tags],
            }
        else:
            breakdown_display = breakdown

        # Return the original tags unchanged
        return tags, breakdown_display

    def _log_breakdown(self, breakdown: dict[str, Any]) -> None:
        """Log the tag breakdown."""
        if breakdown.get("rating"):
            log.info(f"Rating: {breakdown['rating']}")

        if breakdown.get("special"):
            log.info(
                f"Special: {', '.join(breakdown['special'][:5])}{'...' if len(breakdown['special']) > 5 else ''}"
            )

        if breakdown.get("characters"):
            chars = breakdown["characters"]
            log_list_truncated(chars, "Characters")

        if breakdown.get("general"):
            general = breakdown["general"]
            log_list_truncated(general, "General")

    def _format_tags(self, tags: list[str]) -> str:
        """Format tags with delimiter and spacing."""
        if not tags:
            return ""
        if self.use_spaces:
            tags = [t.replace("_", " ") for t in tags]
        return self.delimiter.join(tags)

    def _save_output(self, context: ImageContext, output: str) -> Path:
        """Save output to file and return path."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / f"{context.image_path.stem}{self.suffix}.txt"
        output_path.write_text(output)
        return output_path

    def validate(self, context: ImageContext) -> bool:
        """Run if the section has content."""
        tags = context.get_tags(self.section)
        return bool(tags)

    def process(self, context: ImageContext) -> ImageContext | None:
        """Base process method - should be overridden by subclasses."""
        raise NotImplementedError("Subclasses must implement process()")

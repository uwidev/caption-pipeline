"""
BaseFormatStep: Base class for formatting steps.
"""

from pathlib import Path
from typing import Any

from caption_pipeline.core.context import ImageContext
from caption_pipeline.core.step import PipelineStep
from caption_pipeline.utils.logging_utils import log


class BaseFormatStep(PipelineStep):
    """
    Base class for formatting steps.

    Provides shared functionality:
    - Tag ordering: Rating → Character → General
    - Tag formatting with delimiter and spacing
    - Output saving
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
            section: Section to format (0=prepended, 1=main, 2=NL)
            output_dir: Output directory for the file
            suffix: Suffix to add to the output filename
            delimiter: Delimiter for tags (sections 0 and 1)
            use_spaces: Convert underscores to spaces in tags
        """
        self.section: int = section
        self.output_dir: Path = output_dir or Path("./done/")
        self.suffix: str = suffix
        self.delimiter: str = delimiter
        self.use_spaces: bool = use_spaces

    def _build_ordered_tags(self, context: ImageContext) -> tuple[list[str], dict[str, Any]]:
        """
        Build and order tags for section 1 (main tags).

        Order: Rating → Special Tags → Character Tags → General Tags

        Returns:
            Tuple of (ordered_tags, breakdown)
            - ordered_tags: List of ordered tags in display form
            - breakdown: Dict with 'rating', 'special', 'characters', 'general' counts and previews
        """
        if self.section != 1:
            tags = context.get_tags(self.section)
            return tags, {"count": len(tags), "preview": tags[:5]}

        rating = context.rating
        character_tags = context.get_character_tags()

        # Get the current main tags (characters and rating already removed)
        tags = context.get_tags(1)

        # Start with rating if it exists
        ordered = []
        breakdown = {
            "rating": None,
            "special": [],
            "characters": [],
            "general": [],
        }

        if rating:
            ordered.append(rating)
            breakdown["rating"] = rating

        # Add special tags that exist in the current tags
        SPECIAL_TAGS = {"original", "borrowed_character"}
        special_tags = [tag for tag in tags if tag in SPECIAL_TAGS]
        ordered.extend(special_tags)
        breakdown["special"] = special_tags

        # Add character tags
        if character_tags:
            ordered.extend(character_tags)
            breakdown["characters"] = character_tags

        # Add remaining general tags (everything that isn't special)
        special_set = set(SPECIAL_TAGS)
        general_tags = [tag for tag in tags if tag not in special_set]
        ordered.extend(general_tags)
        breakdown["general"] = general_tags

        # Adjust if spaces
        if self.use_spaces:
            ordered = [tag.replace("_", " ") for tag in ordered]
            # Also update breakdown for display
            if character_tags:
                breakdown["characters"] = [tag.replace("_", " ") for tag in character_tags]
            # General tags already have spaces from the context

        return ordered, breakdown

    def _log_breakdown(self, breakdown: dict[str, Any]) -> None:
        """Log the tag breakdown."""
        if breakdown.get("rating"):
            log.debug(f"  Rating: {breakdown['rating']}")
        
        if breakdown.get("special"):
            log.debug(f"  Special: {', '.join(breakdown['special'][:5])}{'...' if len(breakdown['special']) > 5 else ''}")
        
        if breakdown.get("characters"):
            chars = breakdown['characters']
            log.debug(f"  Characters ({len(chars)}): {', '.join(chars[:5])}{'...' if len(chars) > 5 else ''}")
        
        if breakdown.get("general"):
            general = breakdown['general']
            log.debug(f"  General ({len(general)}): {', '.join(general[:5])}{'...' if len(general) > 5 else ''}")

    def _format_tags(self, tags: list[str]) -> str:
        """Format tags with delimiter and spacing."""
        if not tags:
            return ""
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

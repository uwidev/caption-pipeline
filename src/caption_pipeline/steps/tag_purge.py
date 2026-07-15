"""
TagPurgeStep: Remove all tags except artist and character tags.
"""

from caption_pipeline.core.context import ImageContext
from caption_pipeline.core.help import step_help
from caption_pipeline.core.step import PipelineStep
from caption_pipeline.utils.logging_utils import log, log_list_truncated, section


@step_help(
    name="tag:purge",
    description="Remove all tags except artist and character tags.",
    long_description="""This step removes all tags from sections 0 and 1 (prepended and main)
except for those that are artist or character tags (as stored in context.artists and
context.character_tags). The NL section (section 2) is left untouched.

Useful as a "reset" to start fresh while preserving the identity tags.

You can optionally specify which sections to purge via --section.""",
    options=[
        {
            "flag": "--section {0,1,-1}",
            "help": "Sections to purge: 0=prepended, 1=main, -1=both (default: -1)",
        },
    ],
    example="tag:purge --section 1",
)
class TagPurgeStep(PipelineStep):
    """
    Remove all tags except artist and character tags.
    """

    def __init__(self, target_section: int = -1) -> None:
        """
        Initialize the purge step.

        Args:
            target_section: -1 = both sections 0 and 1, 0 = only prepended, 1 = only main
        """
        self.target_section = target_section

    def name(self) -> str:
        return "tag:purge"

    def validate(self, context: ImageContext) -> bool:
        """Run if there are tags in sections 0 or 1."""
        if self.target_section == -1:
            return bool(context.tags[0] or context.tags[1])
        return bool(context.get_tags(self.target_section))

    def process(self, context: ImageContext) -> ImageContext | None:
        with section(f"Processing: {context.image_path.name}"):
            # Determine which sections to purge
            sections_to_purge = []
            if self.target_section == -1:
                sections_to_purge = [0, 1]
            else:
                sections_to_purge = [self.target_section]

            # Get preserved tags
            artist_set = set(context.get_artists())
            character_set = set(context.get_character_tags())
            preserved_set = artist_set | character_set

            result = context.copy()
            total_removed = 0
            for section_idx in sections_to_purge:
                if section_idx == 2:
                    # NL section: clear it entirely
                    if context.get_tags(2):  # non-empty
                        result.set_tags([], section=2)
                        total_removed += 1  # count as one removal
                        log.debug("Cleared NL section")
                    continue

                tags = context.get_tags(section_idx)
                if not tags:
                    continue

                # Keep only preserved tags
                kept = [tag for tag in tags if tag in preserved_set]
                removed = len(tags) - len(kept)
                total_removed += removed

                # Update the context
                result.set_tags(kept, section_idx)

            if total_removed > 0:
                log.info(
                    f"Removed {total_removed} tags, kept {len(preserved_set)} artist/character tags"
                )
                # Show which tags were kept
                if preserved_set:
                    log_list_truncated(list(preserved_set), "Preserved artist/character tags")
            else:
                log.debug("No tags removed (already only artist/character tags)")

            return result

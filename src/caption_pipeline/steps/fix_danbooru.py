"""
FixDanbooruStep: Fix tags to only keep danbooru tags.
"""

from caption_pipeline.core.context import ImageContext
from caption_pipeline.core.help import step_help
from caption_pipeline.core.step import PipelineStep
from caption_pipeline.utils import log_list_truncated
from caption_pipeline.utils.logging_utils import log, section
from caption_pipeline.utils.tag_db import load_character_tags_only, load_general_tags_only


@step_help(
    name="fix:danbooru",
    description="Fix tags to only keep danbooru tags.",
    long_description="""This step filters tags in section 0 and 1 to only keep tags that exist
in the danbooru tag database. Tags not found in the database are removed.

Character tags and special tags (original, borrowed_character) are preserved
if they exist in the character database.

Whitelisted tags can be specified to always keep certain tags even if they
aren't in the database.

This is useful for ensuring only danbooru-compatible tags are used for training.""",
    options=[
        {
            "flag": "--whitelist TAG,TAG,...",
            "help": "Tags to always keep (overrides danbooru-only filter)",
            "default": "",
        },
        {
            "flag": "--section INT",
            "help": "Section to fix (0=prepended, 1=main, 2=NL, -1=all)",
            "default": "-1",
        },
    ],
    example="fix:danbooru --whitelist 'original,custom_tag'",
)
class FixDanbooruStep(PipelineStep):
    """
    Fix tags to only keep danbooru tags.
    """

    SPECIAL_TAGS = {"original", "borrowed_character"}

    def __init__(
        self,
        whitelist: list[str] | None = None,
        target_section: int = -1,  # Renamed from 'section'
    ) -> None:
        """
        Initialize the danbooru fix step.

        Args:
            whitelist: Tags to always keep (overrides danbooru-only filter)
            target_section: Section to fix (-1 = all, 0 = prepended, 1 = main, 2 = NL)
        """
        self.whitelist: set[str] = set(whitelist or [])
        self.section: int = target_section
        self._general_tags: set[str] | None = None
        self._character_tags: set[str] | None = None

    def name(self) -> str:
        return "fix:danbooru"

    def validate(self, context: ImageContext) -> bool:
        """Run if there are tags to fix."""
        if self.section == -1:
            return bool(context.tags[0] or context.tags[1])
        return bool(context.get_tags(self.section))

    def _load_databases(self) -> None:
        """Lazy load tag databases."""
        if self._general_tags is None:
            self._general_tags = load_general_tags_only()
        if self._character_tags is None:
            self._character_tags = load_character_tags_only()

    def _is_danbooru_tag(self, tag: str) -> bool:
        """
        Check if a tag exists in the danbooru database.

        Tags are normalized (lowercase with underscores).
        """
        self._load_databases()

        # Check if it's a character tag
        if tag in self._character_tags:
            return True

        # Check if it's a general tag
        if tag in self._general_tags:
            return True

        return False

    def _fix_tags(self, tags: list[str]) -> list[str]:
        """Fix a list of tags to only keep danbooru tags."""
        result = []

        for tag in tags:
            # Always keep whitelisted tags
            if tag in self.whitelist:
                result.append(tag)
                continue

            # Keep if it's a danbooru tag
            if self._is_danbooru_tag(tag):
                result.append(tag)
                continue

            # Keep special tags (original, borrowed_character)
            if tag in self.SPECIAL_TAGS:
                result.append(tag)
                continue

        return result

    def process(self, context: ImageContext) -> ImageContext | None:
        """Fix tags to only keep danbooru tags."""
        with section(f"Processing: {context.image_path.name}"):
            result = context.copy()

            # Determine which sections to fix
            sections_to_fix = []
            if self.section == -1:
                sections_to_fix = [0, 1]  # Only fix prepended and main tags
            else:
                sections_to_fix = [self.section]

            original_counts = {}
            fixed_counts = {}
            all_removed_tags = set()

            for section_idx in sections_to_fix:
                # Skip NL section (section 2) as it's not tag-based
                if section_idx == 2:
                    continue

                tags = context.get_tags(section_idx)
                if not tags:
                    continue

                original_counts[section_idx] = len(tags)
                fixed = self._fix_tags(tags)
                fixed_counts[section_idx] = len(fixed)

                # Track removed tags
                removed = set(tags) - set(fixed)
                all_removed_tags.update(removed)

                result.set_tags(fixed, section_idx)

            # Log results
            if original_counts:
                total_original = sum(original_counts.values())
                total_fixed = sum(fixed_counts.values())
                removed = total_original - total_fixed

                log.info(
                    f"Danbooru: {total_original} tags → {total_fixed} tags ({removed} removed)"
                )

                if all_removed_tags and removed > 0:
                    log_list_truncated(list(all_removed_tags), "Removed tags")

            return result

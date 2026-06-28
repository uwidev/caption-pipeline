"""
FilterDanbooruStep: Filter tags to only keep danbooru tags.
"""

from caption_pipeline.core.context import ImageContext
from caption_pipeline.core.help import step_help
from caption_pipeline.core.step import PipelineStep
from caption_pipeline.utils.logging_utils import log
from caption_pipeline.utils.tag_db import load_character_tags_only, load_general_tags_only


@step_help(
    name="filter:danbooru_only",
    description="Filter tags to only keep danbooru tags.",
    long_description="""This step filters tags in section 0 and 1 to only keep tags that exist
in the danbooru tag database. Tags not found in the database are removed.

Character tags and special tags (original, borrowed_character) are preserved
if they exist in the character database.

Whitelisted tags can be specified to always keep certain tags even if they
aren't in the database.""",
    options=[
        {
            "flag": "--whitelist TAG,TAG,...",
            "help": "Tags to always keep (overrides danbooru-only filter)",
            "default": "",
        },
        {
            "flag": "--section INT",
            "help": "Section to filter (0=prepended, 1=main, 2=NL, -1=all)",
            "default": "-1",
        },
    ],
    example="filter:danbooru_only --whitelist 'original,custom_tag'",
)
class FilterDanbooruStep(PipelineStep):
    """
    Filter tags to only keep danbooru tags.
    """

    def __init__(
        self,
        whitelist: list[str] | None = None,
        section: int = 1,  # -1 means all sections
    ) -> None:
        """
        Initialize the danbooru filter step.

        Args:
            whitelist: Tags to always keep (overrides danbooru-only filter)
            section: Section to filter (-1 = all, 0 = prepended, 1 = main, 2 = NL)
        """
        self.whitelist: set[str] = set(whitelist or [])
        self.section: int = section
        self._general_tags: set[str] | None = None
        self._character_tags: set[str] | None = None

    def name(self) -> str:
        return "filter:danbooru_only"

    def validate(self, context: ImageContext) -> bool:
        """Run if there are tags to filter."""
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

    def _filter_tags(self, tags: list[str]) -> list[str]:
        """Filter a list of tags to only keep danbooru tags."""
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

            # Check if it's a special tag (original, borrowed_character)
            # These are valid even if not in the database
            if tag in {"original", "borrowed_character"}:
                result.append(tag)
                continue

        return result

    def process(self, context: ImageContext) -> ImageContext | None:
        """Filter tags to only keep danbooru tags."""
        with log.section(f"Processing: {context.image_path.name}"):
            result = context.copy()

            # Determine which sections to filter
            sections_to_filter = []
            if self.section == -1:
                sections_to_filter = [0, 1]  # Only filter prepended and main tags
            else:
                sections_to_filter = [self.section]

            original_counts = {}
            filtered_counts = {}

            for section in sections_to_filter:
                # Skip NL section (section 2) as it's not tag-based
                if section == 2:
                    continue

                tags = context.get_tags(section)
                if not tags:
                    continue

                original_counts[section] = len(tags)
                filtered = self._filter_tags(tags)
                filtered_counts[section] = len(filtered)
                result.set_tags(filtered, section)

            # Log results
            if original_counts:
                total_original = sum(original_counts.values())
                total_filtered = sum(filtered_counts.values())
                removed = total_original - total_filtered

                log.info(
                    f"Danbooru filter: {total_original} tags → {total_filtered} tags "
                    f"  ({removed} removed)"
                )

                if removed > 0:
                    # Find which tags were removed
                    original_set = set()
                    for section in sections_to_filter:
                        original_set.update(context.get_tags(section))
                    filtered_set = set()
                    for section in sections_to_filter:
                        filtered_set.update(result.get_tags(section))
                    removed_tags = original_set - filtered_set

                    if removed_tags:
                        log.debug(f"Removed tags: {', '.join(sorted(removed_tags))}")

            return result

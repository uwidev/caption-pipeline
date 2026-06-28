"""
FilterOverlapStep: Drop overlapping tags using dghs-imgutils.
"""

from imgutils.tagging import drop_overlap_tags

from caption_pipeline.core.context import ImageContext
from caption_pipeline.core.help import step_help
from caption_pipeline.core.step import PipelineStep
from caption_pipeline.utils.logging_utils import log


@step_help(
    name="filter:drop_overlap",
    description="Drop overlapping tags using dghs-imgutils.",
    long_description="""This step removes tags that have overlaps with other tags based on 
precomputed overlap information from dghs-imgutils.

For example: 'long_hair' and 'very_long_hair' overlap, so only 'very_long_hair' is kept.
'breasts' and 'medium_breasts' overlap, so only 'medium_breasts' is kept.

The step can operate on either:
- A list of tags (removes overlapping tags)
- A dict of tag->confidence (removes overlapping tags and returns dict)

This is useful for cleaning up tag lists by removing redundant tags.""",
    options=[
        {
            "flag": "--section INT",
            "help": "Section to filter (0=prepended, 1=main, 2=NL, -1=all)",
            "default": "-1",
        },
        {
            "flag": "--keep-scored",
            "help": "When true, keep tag scores and return a dict. Otherwise return list.",
            "default": "False",
        },
    ],
    example="filter:drop_overlap --section 1",
)
class FilterOverlapStep(PipelineStep):
    """
    Drop overlapping tags using dghs-imgutils.
    """

    def __init__(
        self,
        section: int = -1,  # -1 means all sections
        keep_scored: bool = False,
    ) -> None:
        """
        Initialize the drop overlap filter step.

        Args:
            section: Section to filter (-1 = all, 0 = prepended, 1 = main, 2 = NL)
            keep_scored: Whether to keep tag scores (requires inferenced_tags)
        """
        self.section: int = section
        self.keep_scored: bool = keep_scored

    def name(self) -> str:
        return "filter:drop_overlap"

    def validate(self, context: ImageContext) -> bool:
        """Run if there are tags to filter."""
        if self.section == -1:
            return bool(context.tags[0] or context.tags[1])
        return bool(context.get_tags(self.section))

    def _filter_tags(self, tags: list[str]) -> list[str]:
        """Filter a list of tags by dropping overlaps."""
        if not tags:
            return tags
        
        # Remove duplicates first (preserve order)
        seen = set()
        unique_tags = []
        for tag in tags:
            if tag not in seen:
                seen.add(tag)
                unique_tags.append(tag)
        
        # Drop overlaps using imgutils
        return drop_overlap_tags(unique_tags)

    def _filter_scored_tags(self, tags: dict[str, float]) -> dict[str, float]:
        """Filter a dict of scored tags by dropping overlaps."""
        if not tags:
            return tags
        
        # Drop overlaps using imgutils (returns dict)
        return drop_overlap_tags(tags)

    def process(self, context: ImageContext) -> ImageContext | None:
        """Drop overlapping tags."""
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
                
                # If we're keeping scores and have inferenced_tags, use scored filtering
                if self.keep_scored and context.inferenced_tags:
                    # Filter scored tags
                    scored_filtered = self._filter_scored_tags(context.inferenced_tags)
                    # Then filter the tag list based on the result
                    filtered = [tag for tag in tags if tag in scored_filtered]
                else:
                    filtered = self._filter_tags(tags)
                
                filtered_counts[section] = len(filtered)
                result.set_tags(filtered, section)
            
            # Log results
            if original_counts:
                total_original = sum(original_counts.values())
                total_filtered = sum(filtered_counts.values())
                removed = total_original - total_filtered
                
                log.info(
                    f"Drop overlap: {total_original} tags → {total_filtered} tags "
                    f"({removed} removed)"
                )
                
                if removed > 0 and total_original > 0:
                    # Show which tags were kept/removed at DEBUG level
                    if total_original <= 50:  # Only show details for reasonable sizes
                        original_set = set()
                        for section in sections_to_filter:
                            original_set.update(context.get_tags(section))
                        filtered_set = set()
                        for section in sections_to_filter:
                            filtered_set.update(result.get_tags(section))
                        removed_tags = original_set - filtered_set
                        
                        if removed_tags:
                            log.debug(f"Removed tags: {', '.join(sorted(removed_tags)[:10])}")
                            if len(removed_tags) > 10:
                                log.debug(f"... and {len(removed_tags) - 10} more")
            
            return result

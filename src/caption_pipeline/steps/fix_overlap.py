"""
FixOverlapStep: Fix overlapping tags using dghs-imgutils.
"""

from imgutils.tagging import drop_overlap_tags

from caption_pipeline.core.context import ImageContext
from caption_pipeline.core.help import step_help
from caption_pipeline.core.step import PipelineStep
from caption_pipeline.utils.logging_utils import log, log_truncated, section


@step_help(
    name="fix:overlap",
    description="Fix overlapping tags using dghs-imgutils.",
    long_description="""This step removes tags that have overlaps with other tags based on 
precomputed overlap information from dghs-imgutils.

For example: 'long_hair' and 'very_long_hair' overlap, so only 'very_long_hair' is kept.
'breasts' and 'medium_breasts' overlap, so only 'medium_breasts' is kept.

The step can operate on either:
- A list of tags (removes overlapping tags)
- A dict of tag->confidence (removes overlapping tags and returns dict)

Use --keep-hints to preserve original hinted tags even if they would be dropped.

This is useful for cleaning up tag lists by removing redundant tags.""",
    options=[
        {
            "flag": "--section INT",
            "help": "Section to fix (0=prepended, 1=main, 2=NL, -1=all)",
            "default": "-1",
        },
        {
            "flag": "--keep-scored",
            "help": "When true, keep tag scores and return a dict. Otherwise return list.",
            "default": "False",
        },
        {
            "flag": "--keep-hints",
            "help": "Preserve original hinted tags even if they would be dropped",
            "default": "False",
        },
    ],
    example="fix:overlap --section 1 --keep-hints",
)
class FixOverlapStep(PipelineStep):
    """
    Fix overlapping tags using dghs-imgutils.
    """

    def __init__(
        self,
        target_section: int = -1,  # Renamed from 'section'
        keep_scored: bool = False,
        keep_hints: bool = False,
    ) -> None:
        """
        Initialize the fix overlap step.

        Args:
            target_section: Section to fix (-1 = all, 0 = prepended, 1 = main, 2 = NL)
            keep_scored: Whether to keep tag scores (requires inferenced_tags)
            keep_hints: Whether to preserve original hinted tags
        """
        self.section: int = target_section
        self.keep_scored: bool = keep_scored
        self.keep_hints: bool = keep_hints

    def name(self) -> str:
        return "fix:overlap"

    def validate(self, context: ImageContext) -> bool:
        """Run if there are tags to fix."""
        if self.section == -1:
            return bool(context.tags[0] or context.tags[1])
        return bool(context.get_tags(self.section))

    def _fix_tags(self, tags: list[str]) -> list[str]:
        """Fix a list of tags by dropping overlaps."""
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

    def _fix_scored_tags(self, tags: dict[str, float]) -> dict[str, float]:
        """Fix a dict of scored tags by dropping overlaps."""
        if not tags:
            return tags
        
        # Drop overlaps using imgutils (returns dict)
        return drop_overlap_tags(tags)

    def process(self, context: ImageContext) -> ImageContext | None:
        """Fix overlapping tags."""
        with section(f"Processing: {context.image_path.name}"):
            result = context.copy()
            
            # Get original tags for preservation if keep_hints is enabled
            original_set = set(context.get_original_flat()) if self.keep_hints else set()
            
            if self.keep_hints:
                log.debug(f"Preserving original hinted tags ({len(original_set)} total)")
            
            # Determine which sections to fix
            sections_to_fix = []
            if self.section == -1:
                sections_to_fix = [0, 1]  # Only fix prepended and main tags
            else:
                sections_to_fix = [self.section]
            
            original_counts = {}
            fixed_counts = {}
            all_removed_tags = set()
            all_preserved_tags = set()
            
            for section_idx in sections_to_fix:
                # Skip NL section (section 2) as it's not tag-based
                if section_idx == 2:
                    continue
                
                tags = context.get_tags(section_idx)
                if not tags:
                    continue
                
                original_counts[section_idx] = len(tags)
                
                # If we're keeping scores and have inferenced_tags, use scored fixing
                if self.keep_scored and context.inferenced_tags:
                    # Fix scored tags
                    scored_fixed = self._fix_scored_tags(context.inferenced_tags)
                    # Then filter the tag list based on the result
                    fixed = [tag for tag in tags if tag in scored_fixed]
                else:
                    fixed = self._fix_tags(tags)
                
                # Preserve original hinted tags if enabled
                if self.keep_hints and original_set:
                    removed = set(tags) - set(fixed)
                    original_removed = removed & original_set
                    
                    if original_removed:
                        # Add back original tags that were dropped
                        fixed = list(set(fixed) | original_removed)
                        all_preserved_tags.update(original_removed)
                
                fixed_counts[section_idx] = len(fixed)
                
                # Track all removed tags (for logging)
                if original_counts[section_idx] > fixed_counts[section_idx]:
                    removed = set(tags) - set(fixed)
                    all_removed_tags.update(removed)
                
                result.set_tags(fixed, section_idx)
            
            # Log results
            if original_counts:
                total_original = sum(original_counts.values())
                total_fixed = sum(fixed_counts.values())
                removed = total_original - total_fixed
                
                # Build the log message with keep-hints info
                log_msg = f"Overlap: {total_original} tags → {total_fixed} tags ({removed} removed)"
                if self.keep_hints and all_preserved_tags:
                    log_msg += f" ({len(all_preserved_tags)} original hints preserved)"
                log.info(log_msg)
                
                # DEBUG: Show all preserved tags
                if self.keep_hints and all_preserved_tags:
                    log.debug(f"Preserved original tags ({len(all_preserved_tags)}): {', '.join(sorted(all_preserved_tags))}")
                
                if all_removed_tags and removed > 0:
                    tags_str = ", ".join(sorted(all_removed_tags)[:10])
                    log_truncated(f"Removed tags ({len(all_removed_tags)})", tags_str, max_len=64)
            
            return result

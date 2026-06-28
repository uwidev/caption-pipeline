"""
FixCountsStep: Resolve conflicting count tags.

This step resolves conflicting count tags (1girl, 2girls, 3girls, etc.)
by keeping the highest confidence tag per category.
"""

from caption_pipeline.core.context import ImageContext
from caption_pipeline.core.help import step_help
from caption_pipeline.core.step import PipelineStep
from caption_pipeline.utils.logging_utils import log, log_truncated, section


@step_help(
    name="fix:counts",
    description="Resolve conflicting count tags.",
    long_description="""This step resolves conflicting count tags by keeping the highest
confidence tag per category.

For example:
- If '1girl' (0.95) and '3girls' (0.60) are present, '1girl' is kept.
- If 'solo' is present, all count tags are dropped (solo overrides counts).

Count tags are resolved per category:
- boys: 1boy, 2boys, 3boys, 4boys, 5boys, multiple boys
- girls: 1girl, 2girls, 3girls, 4girls, 5girls, multiple girls
- others: 1other, 2others, 3others, 4others, 5others, multiple others

This step should be run after tag generation and before format:join.

Note: Tags are assumed to be in confidence order (highest first).""",
    options=[
        {
            "flag": "--section INT",
            "help": "Section to resolve (0=prepended, 1=main, -1=all)",
            "default": "1",
        },
    ],
    example="fix:counts --section 1",
)
class FixCountsStep(PipelineStep):
    """
    Resolve conflicting count tags.
    """

    COUNT_CATEGORIES = {
        "boys": ["1boy", "2boys", "3boys", "4boys", "5boys", "multiple boys"],
        "girls": ["1girl", "2girls", "3girls", "4girls", "5girls", "multiple girls"],
        "others": ["1other", "2others", "3others", "4others", "5others", "multiple others"],
    }

    def __init__(
        self,
        target_section: int = 1,  # Renamed from 'section'
    ) -> None:
        """
        Initialize the fix counts step.

        Args:
            target_section: Section to resolve (0=prepended, 1=main, -1=all)
        """
        self.section: int = target_section

    def name(self) -> str:
        return "fix:counts"

    def validate(self, context: ImageContext) -> bool:
        """Run if there are tags to resolve."""
        if self.section == -1:
            return bool(context.tags[0] or context.tags[1])
        return bool(context.get_tags(self.section))

    def _resolve_counts(self, tags: list[str]) -> list[str]:
        """
        Resolve conflicting count tags.

        Tags are assumed to be in confidence order (highest first).
        Keeps the first (highest confidence) count tag per category.
        """
        resolved = []
        seen_categories = set()
        has_solo = False

        for tag in tags:
            # Check if solo is present
            if tag == "solo":
                has_solo = True
                resolved.append(tag)
                continue

            # Check if this is a count tag
            found_category = None
            for category, count_tags in self.COUNT_CATEGORIES.items():
                if tag in count_tags:
                    found_category = category
                    break

            if found_category:
                # If solo is present, drop all count tags
                if has_solo:
                    continue
                # Only keep the first (highest confidence) count tag per category
                if found_category not in seen_categories:
                    seen_categories.add(found_category)
                    resolved.append(tag)
                # Else skip this tag (already have a count for this category)
            else:
                resolved.append(tag)

        return resolved

    def process(self, context: ImageContext) -> ImageContext | None:
        """Resolve conflicting count tags."""
        with section(f"Processing: {context.image_path.name}"):
            result = context.copy()

            # Determine which sections to process
            sections_to_process = []
            if self.section == -1:
                sections_to_process = [0, 1]  # Only prepended and main tags
            else:
                sections_to_process = [self.section]

            original_counts = {}
            resolved_counts = {}
            all_removed_tags = set()

            for section_idx in sections_to_process:
                # Skip NL section (section 2)
                if section_idx == 2:
                    continue

                tags = context.get_tags(section_idx)
                if not tags:
                    continue

                original_counts[section_idx] = len(tags)
                resolved = self._resolve_counts(tags)
                resolved_counts[section_idx] = len(resolved)

                # Track removed tags
                removed = set(tags) - set(resolved)
                all_removed_tags.update(removed)

                result.set_tags(resolved, section_idx)

            # Log results
            if original_counts:
                total_original = sum(original_counts.values())
                total_resolved = sum(resolved_counts.values())
                removed = total_original - total_resolved

                log.info(
                    f"Counts: {total_original} tags → {total_resolved} tags "
                    f"({removed} removed)"
                )

                if all_removed_tags and removed > 0:
                    tags_str = ", ".join(sorted(all_removed_tags)[:10])
                    log_truncated(f"Removed count tags ({len(all_removed_tags)})", tags_str, max_len=64)

            return result

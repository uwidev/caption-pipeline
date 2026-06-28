"""
FormatJoinStep: Join tag sections into final caption format.
"""

from pathlib import Path

from caption_pipeline.core.context import ImageContext
from caption_pipeline.core.help import step_help
from caption_pipeline.steps.format_base import BaseFormatStep
from caption_pipeline.utils.logging_utils import (
    log,
    log_truncated,
    section,
)


def normalize_tag_for_comparison(tag: str) -> str:
    """
    Normalize a tag for comparison purposes.
    """
    tag = tag.lower().strip()
    if tag.startswith("character:"):
        tag = tag[10:].strip()
    tag = tag.replace("_", " ")
    tag = " ".join(tag.split())
    return tag


@step_help(
    name="format:join",
    description="Join tag sections with delimiters.",
    long_description="""This step combines all tag sections into the final caption format.

Key operations:
1. Orders tags as: Rating → Special Tags → Character Tags → General Tags
2. Converts underscores to spaces for readability
3. Joins sections with the configured delimiter
4. Saves the final caption to disk

Note: All fixing (counts, overlaps, danbooru) should be done before this step.""",
    options=[
        {"flag": "--delimiter TEXT", "help": "Delimiter between sections", "default": " ||| "},
        {"flag": "--output-dir PATH", "help": "Output directory", "default": "./done/"},
        {"flag": "--tag-suffix TEXT", "help": "Suffix for tag files", "default": ""},
        {"flag": "--no-deduplicate", "help": "Don't deduplicate tags", "default": "deduplicate"},
        {"flag": "--no-spaces", "help": "Keep underscores in tags", "default": "use spaces"},
    ],
    example="format:join --delimiter ' ||| ' --output-dir ./done/",
)
class FormatJoinStep(BaseFormatStep):
    """
    Join tag sections into final caption format.
    """

    def __init__(
        self,
        delimiter: str = " ||| ",
        output_dir: Path | None = None,
        tag_suffix: str = "",
        deduplicate_tags: bool = True,
        use_spaces: bool = True,
    ):
        """Initialize the format join step."""
        super().__init__(
            section=1,
            output_dir=output_dir,
            suffix=tag_suffix,
            delimiter=", ",
            use_spaces=use_spaces,
        )
        self.section_delimiter = delimiter
        self.deduplicate_tags = deduplicate_tags

    def name(self) -> str:
        return "format:join"

    def validate(self, context: ImageContext) -> bool:
        """Run if there are tags to format."""
        if context.tags[0] or context.tags[1] or context.tags[2]:
            return True
        if context.has_characters():
            return True
        return False

    def _deduplicate_tags_preserve_order(self, tags: list[str]) -> list[str]:
        """Remove duplicates while preserving order."""
        seen = set()
        result = []

        for tag in tags:
            normalized = normalize_tag_for_comparison(tag)
            if normalized not in seen:
                seen.add(normalized)
                result.append(tag)

        return result

    def _format_section(self, tags: list[str]) -> tuple[str, dict]:
        """
        Format a section's tags with delimiter and spacing.

        Returns:
            Tuple of (formatted_string, breakdown)
        """
        if not tags:
            return "", {"count": 0, "preview": []}

        original_count = len(tags)

        if self.deduplicate_tags:
            tags = self._deduplicate_tags_preserve_order(tags)

        if self.use_spaces:
            tags = [tag.replace("_", " ") for tag in tags]

        formatted = ", ".join(tags)

        return formatted, {
            "original_count": original_count,
            "count": len(tags),
            "preview": tags[:5],
        }

    def process(self, context: ImageContext) -> ImageContext | None:
        """Join and save the caption."""
        with section(f"Processing: {context.image_path.name}"):
            # === SECTION 0: Prepended tags ===
            section0, breakdown0 = self._format_section(context.tags[0])
            if breakdown0["count"] > 0:
                log.debug(
                    f"  Prepended ({breakdown0['count']}): {', '.join(breakdown0['preview'])}{'...' if breakdown0['count'] > 5 else ''}"
                )

            # === SECTION 1: Main tags ===
            main_tags, breakdown1 = self._build_ordered_tags(context)
            section1, _ = self._format_section(main_tags)

            # Log breakdown from _build_ordered_tags
            self._log_breakdown(breakdown1)

            # === SECTION 2: NL caption ===
            if context.tags[2]:
                if len(context.tags[2]) == 1:
                    section2 = context.tags[2][0]
                else:
                    section2 = "\n".join(context.tags[2])
                log_truncated("NL", section2, max_len=64, level="info", continuation_level="debug")
            else:
                section2 = ""

            sections = [section0, section1, section2]

            # Skip if all sections are empty
            if not any(sections):
                log.debug("All sections empty - skipping save")
                return context

            # Join sections with delimiter
            caption = self.section_delimiter.join(sections)

            # Save to disk
            self.output_dir.mkdir(parents=True, exist_ok=True)
            output_path = self.output_dir / f"{context.image_path.stem}{self.suffix}.txt"
            output_path.write_text(caption)

            # Log output with truncation
            log_truncated("Written", caption, max_len=64, level="info", continuation_level="debug")

            # Store result
            result = context.copy()
            result.metadata["caption"] = caption
            return result

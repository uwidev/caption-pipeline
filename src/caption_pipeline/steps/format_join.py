"""
FormatJoinStep: Join tag sections into final caption format.
"""

import re
from pathlib import Path

from caption_pipeline.core.context import ImageContext
from caption_pipeline.core.help import step_help
from caption_pipeline.steps.format_base import BaseFormatStep
from caption_pipeline.utils.logging_utils import log


# Count tag patterns
_COUNT_PATTERNS = {
    "boys": {
        "counts": {1: "1boy", 2: "2boys", 3: "3boys", 4: "4boys", 5: "5boys"},
        "multiple": "multiple boys",
        "single": "1boy",
    },
    "girls": {
        "counts": {1: "1girl", 2: "2girls", 3: "3girls", 4: "4girls", 5: "5girls"},
        "multiple": "multiple girls",
        "single": "1girl",
    },
    "others": {
        "counts": {1: "1other", 2: "2others", 3: "3others", 4: "4others", 5: "5others"},
        "multiple": "multiple others",
        "single": "1other",
    },
}


def normalize_tag_for_comparison(tag: str) -> str:
    """
    Normalize a tag for comparison purposes.

    - Converts to lowercase
    - Replaces underscores with spaces
    - Strips whitespace
    - Removes "character:" prefix if present
    """
    tag = tag.lower().strip()
    if tag.startswith("character:"):
        tag = tag[10:].strip()
    tag = tag.replace("_", " ")
    tag = " ".join(tag.split())
    return tag


@step_help(
    name="format:join",
    description="Join tag sections with delimiters and add character tags back.",
    long_description="""This step combines all tag sections into the final caption format.

Key operations:
1. Orders tags as: Rating → Character tags → General tags (for section 1)
2. Adds character tags from context.character_tags back to section 1 (main tags)
3. Converts underscores to spaces for readability
4. Deduplicates and cleans tags
5. Joins sections with the configured delimiter
6. Saves the final caption to disk

Character tags are removed during processing (for ToriiGate grounding) but MUST
be added back in the final output for training data completeness.
""",
    options=[
        {"flag": "--delimiter TEXT", "help": "Delimiter between sections", "default": " ||| "},
        {"flag": "--output-dir PATH", "help": "Output directory", "default": "./done/"},
        {"flag": "--tag-suffix TEXT", "help": "Suffix for tag files", "default": ""},
        {"flag": "--no-deduplicate", "help": "Don't deduplicate tags", "default": "deduplicate"},
        {"flag": "--no-clean", "help": "Don't clean tags", "default": "clean"},
        {"flag": "--save-empty", "help": "Save even if caption is empty", "default": "don't save"},
        {
            "flag": "--no-resolve-counts",
            "help": "Don't resolve count tags (1boy, 2girls, etc.)",
            "default": "resolve",
        },
        {
            "flag": "--no-character-tags",
            "help": "Don't add character tags to main tags",
            "default": "add characters",
        },
        {
            "flag": "--no-spaces",
            "help": "Keep underscores in tags (don't convert to spaces)",
            "default": "use spaces",
        },
    ],
    example="format:join --delimiter ' ||| ' --output-dir ./done/",
)
class FormatJoinStep(BaseFormatStep):
    """
    Join tag sections into final caption format.

    This step adds character tags back to the main tag list because they were
    removed during processing for ToriiGate grounding. In the final output,
    character names should be in the tag list for training data completeness.
    """

    def __init__(
        self,
        delimiter: str = " ||| ",
        output_dir: Path | None = None,
        save_tags: bool = True,
        tag_suffix: str = "",
        deduplicate_tags: bool = True,
        clean_tags: bool = True,
        save_empty: bool = False,
        resolve_counts: bool = True,
        include_character_tags: bool = True,
        use_spaces: bool = True,
    ):
        """
        Initialize the format join step.

        Args:
            delimiter: Delimiter between sections (default: " ||| ")
            output_dir: Output directory for captions (default: ./done/)
            save_tags: Whether to save tags to disk (default: True)
            tag_suffix: Suffix for tag files (default: "")
            deduplicate_tags: Remove duplicate tags (default: True)
            clean_tags: Clean and normalize tags (default: True)
            save_empty: Save even if caption is empty (default: False)
            resolve_counts: Resolve count tags (1boy, 2girls) (default: True)
            include_character_tags: Add character tags to main tags (default: True)
            use_spaces: Convert underscores to spaces (default: True)
        """
        # Initialize base with section 1 (main tags)
        super().__init__(
            section=1,
            output_dir=output_dir,
            suffix=tag_suffix,
            delimiter=", ",  # Inner delimiter for tags within a section
            use_spaces=use_spaces,
        )
        self.section_delimiter = delimiter
        self.save_tags = save_tags
        self.deduplicate_tags = deduplicate_tags
        self.clean_tags = clean_tags
        self.save_empty = save_empty
        self.resolve_counts = resolve_counts
        self.include_character_tags = include_character_tags

    def name(self) -> str:
        return "format:join"

    def validate(self, context: ImageContext) -> bool:
        """Run if there are tags to format or save_empty is True."""
        if self.save_empty:
            return True
        if context.tags[0] or context.tags[1] or context.tags[2]:
            return True
        if context.has_characters():
            return True
        return False

    def _clean_tag(self, tag: str) -> str:
        """Clean a single tag."""
        if not tag:
            return ""

        cleaned = tag.strip()
        cleaned = cleaned.strip('.,;:!?"\'')
        cleaned = ' '.join(cleaned.split())
        cleaned = cleaned.replace(',', '')
        return cleaned

    def _deduplicate_tags_preserve_order(self, tags: list[str]) -> list[str]:
        """Remove duplicates while preserving order."""
        seen = set()
        result = []

        for tag in tags:
            normalized = normalize_tag_for_comparison(tag)
            if normalized not in seen:
                seen.add(normalized)
                result.append(tag)
            else:
                log.debug(f"Removed duplicate tag: '{tag}'")

        return result

    def _resolve_count_tags(self, tags: list[str]) -> list[str]:
        """Resolve count tags to a single tag per category."""
        tag_dict = {tag: 1.0 for tag in tags}
        result = tag_dict.copy()
        to_remove: set[str] = set()
        solo_present = "solo" in result

        for category, pattern in _COUNT_PATTERNS.items():
            count_tags = pattern["counts"].values()
            multiple_tag = pattern["multiple"]
            single_tag = pattern["single"]

            present_counts = [tag for tag in count_tags if tag in result]
            multiple_present = multiple_tag in result

            if not present_counts and not multiple_present:
                continue

            if solo_present:
                if single_tag in result:
                    for tag in present_counts:
                        if tag != single_tag:
                            to_remove.add(tag)
                else:
                    for tag in present_counts:
                        to_remove.add(tag)
                if multiple_present:
                    to_remove.add(multiple_tag)
                continue

            if len(present_counts) == 1 and not multiple_present:
                continue

            if present_counts:
                def get_count(tag: str) -> int:
                    match = re.match(r'^(\d+)', tag)
                    return int(match.group(1)) if match else 0

                highest_count_tag = max(present_counts, key=get_count)
                for tag in present_counts:
                    if tag != highest_count_tag:
                        to_remove.add(tag)

            if multiple_present and present_counts:
                to_remove.add(multiple_tag)

        for tag in to_remove:
            if tag in result:
                del result[tag]

        if to_remove and self.resolve_counts:
            log.debug(f"Removed {len(to_remove)} duplicate count tags: {', '.join(sorted(to_remove))}")

        return [tag for tag in tags if tag in result]

    def _clean_and_process_tags(self, tags: list[str]) -> list[str]:
        """Clean, deduplicate, and resolve count tags."""
        result = []
        seen = set()

        for tag in tags:
            if ',' in tag:
                parts = [p.strip() for p in tag.split(',') if p.strip()]
                for part in parts:
                    cleaned = self._clean_tag(part)
                    if cleaned:
                        normalized = normalize_tag_for_comparison(cleaned)
                        if normalized not in seen:
                            seen.add(normalized)
                            result.append(cleaned)
            else:
                cleaned = self._clean_tag(tag)
                if cleaned:
                    normalized = normalize_tag_for_comparison(cleaned)
                    if normalized not in seen:
                        seen.add(normalized)
                        result.append(cleaned)

        if self.resolve_counts:
            result = self._resolve_count_tags(result)

        return result

    def _get_character_tags_to_add(self, context: ImageContext) -> list[str]:
        """Get character tags that should be added to the main tag list."""
        if not self.include_character_tags or not context.has_characters():
            return []

        existing_tags = context.tags[0] + context.tags[1]
        existing_normalized = {normalize_tag_for_comparison(t) for t in existing_tags}

        char_tags = context.get_character_tags()
        tags_to_add = []

        for tag in char_tags:
            tag_with_spaces = tag.replace("_", " ")
            normalized = normalize_tag_for_comparison(tag_with_spaces)

            if normalized not in existing_normalized:
                final_tag = tag_with_spaces if self.use_spaces else tag
                tags_to_add.append(final_tag)
                log.debug(f"Adding character tag: '{tag}' -> '{final_tag}'")
            else:
                log.debug(f"Skipping duplicate character tag: '{tag}'")

        return tags_to_add

    def process(self, context: ImageContext) -> ImageContext | None:
        """Join and save the caption."""
        with log.section(f"Processing: {context.image_path.name}"):
            sections = []

            # === SECTION 0: Prepended tags ===
            if context.tags[0]:
                if self.clean_tags:
                    cleaned_section = self._clean_and_process_tags(context.tags[0])
                else:
                    cleaned_section = context.tags[0]
                if self.use_spaces:
                    cleaned_section = [tag.replace("_", " ") for tag in cleaned_section]
                sections.append(", ".join(cleaned_section))
            else:
                sections.append("")

            # === SECTION 1: Main tags ===
            main_tags = context.get_tags(section=1).copy()

            # Add character tags back
            character_tags = self._get_character_tags_to_add(context)
            if character_tags:
                main_tags.extend(character_tags)
                log.debug(f"Added {len(character_tags)} character tags")

            # Clean and process
            if self.clean_tags:
                main_tags = self._clean_and_process_tags(main_tags)

            if self.deduplicate_tags:
                main_tags = self._deduplicate_tags_preserve_order(main_tags)

            # Order tags: Rating → Character → General
            main_tags = self._order_tags(main_tags, context)

            if self.use_spaces:
                main_tags = [tag.replace("_", " ") for tag in main_tags]

            if main_tags:
                sections.append(", ".join(main_tags))
            else:
                sections.append("")

            # === SECTION 2: NL caption ===
            if context.tags[2]:
                if len(context.tags[2]) == 1:
                    sections.append(context.tags[2][0])
                else:
                    sections.append("\n".join(context.tags[2]))
            else:
                sections.append("")

            # Ensure 3 sections
            while len(sections) < 3:
                sections.append("")

            # Skip if empty
            if not any(sections) and not self.save_empty:
                log.debug("All sections empty - skipping save")
                return context

            # Join and save
            caption = self.section_delimiter.join(sections)

            self.output_dir.mkdir(parents=True, exist_ok=True)

            if self.save_tags:
                output_path = self.output_dir / f"{context.image_path.stem}{self.suffix}.txt"
                output_path.write_text(caption)
                log.debug(f"Saved caption to {output_path}")

            # Store result
            result = context.copy()
            result.metadata["caption"] = caption
            return result

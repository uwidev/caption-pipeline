"""
FormatJoinStep: Join tag sections into final caption format.
"""

from pathlib import Path

from caption_pipeline.core.context import ImageContext
from caption_pipeline.core.help import step_help
from caption_pipeline.core.step import PipelineStep
from caption_pipeline.utils import log_list_truncated
from caption_pipeline.utils.logging_utils import log, log_truncated, section


def normalize_tag_for_comparison(tag: str) -> str:
    tag = tag.lower().strip()
    if tag.startswith("character:"):
        tag = tag[10:].strip()
    tag = tag.replace("_", " ")
    tag = " ".join(tag.split())
    return tag


@step_help(
    name="format:join",
    description="Join tag sections with delimiters and save to file.",
    long_description="""This step combines all tag sections into the final caption format.

It does NOT reorder tags – it uses the tags as they are in the context.
If you want ordering, use fix:order before this step.

Operations:
- Optionally deduplicates tags (case‑insensitive, ignoring 'character:' prefix).
- Converts underscores to spaces for readability (optional).
- Joins sections with the configured delimiter.
- Saves the final caption to disk.

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
class FormatJoinStep(PipelineStep):
    def __init__(
        self,
        delimiter: str = " ||| ",
        output_dir: Path | None = None,
        tag_suffix: str = "",
        deduplicate_tags: bool = True,
        use_spaces: bool = True,
    ):
        self.delimiter = delimiter
        self.output_dir = output_dir or Path("./done/")
        self.tag_suffix = tag_suffix
        self.deduplicate_tags = deduplicate_tags
        self.use_spaces = use_spaces

    def name(self) -> str:
        return "format:join"

    def validate(self, context: ImageContext) -> bool:
        return any(context.tags[0]) or any(context.tags[1]) or any(context.tags[2])

    def _deduplicate_tags(self, tags: list[str]) -> list[str]:
        """Remove duplicates while preserving order."""
        seen = set()
        result = []
        for tag in tags:
            norm = normalize_tag_for_comparison(tag)
            if norm not in seen:
                seen.add(norm)
                result.append(tag)
        return result

    def _format_section(self, tags: list[str]) -> tuple[str, list[str]]:
        if not tags:
            return "", []
        if self.deduplicate_tags:
            tags = self._deduplicate_tags(tags)
        if self.use_spaces:
            tags = [tag.replace("_", " ") for tag in tags]
        return ", ".join(tags), tags

    def process(self, context: ImageContext) -> ImageContext | None:
        with section(f"Processing: {context.image_path.name}"):
            # Prepare each section as a string (empty string if no content)
            sections_str = ["", "", ""]

            # SECTION 0: Prepended tags
            if context.tags[0]:
                section0, breakdown0 = self._format_section(context.tags[0])
                sections_str[0] = section0
                log_list_truncated(breakdown0, "Prepended")

            # SECTION 1: Main tags
            if context.tags[1]:
                section1, breakdown1 = self._format_section(context.tags[1])
                sections_str[1] = section1
                log_list_truncated(breakdown1, "Main")

            # SECTION 2: NL caption
            if context.tags[2]:
                if len(context.tags[2]) == 1:
                    section2 = context.tags[2][0]
                else:
                    section2 = "\n".join(context.tags[2])
                sections_str[2] = section2
                log_truncated("NL", section2, max_len=64, level="info", continuation_level="debug")

            # Find the last non‑empty section index
            last_non_empty = -1
            for i in range(2, -1, -1):
                if sections_str[i]:
                    last_non_empty = i
                    break

            if last_non_empty == -1:
                log.debug("All sections empty - skipping save")
                return context

            # Join sections from 0 to last_non_empty inclusive
            caption = self.delimiter.join(sections_str[:last_non_empty + 1])

            self.output_dir.mkdir(parents=True, exist_ok=True)
            output_path = self.output_dir / f"{context.image_path.stem}{self.tag_suffix}.txt"
            output_path.write_text(caption)

            log_truncated("Written", caption, max_len=64, level="info", continuation_level="debug")

            result = context.copy()
            result.metadata["caption"] = caption
            return result

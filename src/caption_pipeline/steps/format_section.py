"""
FormatSectionStep: Output only a specific section from the context.
"""

from pathlib import Path

from caption_pipeline.core.context import ImageContext
from caption_pipeline.core.help import step_help
from caption_pipeline.steps.format_base import BaseFormatStep
from caption_pipeline.utils.logging_utils import log, log_truncated


@step_help(
    name="format:section",
    description="Output only a specific section from the context.",
    long_description="""This step outputs a single section from the context to a file.

Sections:
- Section 0: Prepended tags (delimited by delimiter)
- Section 1: Main tags (delimited by delimiter) - automatically includes:
  - Rating (if present)
  - Special tags (original, borrowed_character)
  - Character tags (added back from context.character_tags)
  - General tags (everything else)
- Section 2: Natural language caption (raw text, delimiter ignored)

The order for section 1 is: Rating → Special Tags → Character Tags → General Tags

This is useful for extracting just the NL caption or just the tags for
further processing or validation.""",
    options=[
        {
            "flag": "--section INT",
            "help": "Section to output (0=prepended, 1=main, 2=NL)",
            "default": "1",
        },
        {
            "flag": "--output-dir PATH",
            "help": "Output directory for the file",
            "default": "./done/",
        },
        {
            "flag": "--suffix TEXT",
            "help": "Suffix to add to the output filename",
            "default": "",
        },
        {
            "flag": "--delimiter TEXT",
            "help": "Delimiter for tags (sections 0 and 1)",
            "default": ", ",
        },
        {
            "flag": "--use-spaces",
            "help": "Convert underscores to spaces in tags",
            "default": "True",
        },
        {
            "flag": "--no-use-spaces",
            "help": "Keep underscores in tags",
            "default": "False",
        },
    ],
    example="format:section --section 1 --delimiter ',' --suffix -tags",
)
class FormatSectionStep(BaseFormatStep):
    """
    Output only a specific section from the context.
    """

    def __init__(
        self,
        section: int = 1,
        output_dir: Path | None = None,
        suffix: str = "",
        delimiter: str = ", ",
        use_spaces: bool = True,
    ) -> None:
        """Initialize the format section step."""
        super().__init__(
            section=section,
            output_dir=output_dir,
            suffix=suffix,
            delimiter=delimiter,
            use_spaces=use_spaces,
        )

    def name(self) -> str:
        return "format:section"

    def process(self, context: ImageContext) -> ImageContext | None:
        """Output the specified section."""
        with log.section(f"Processing: {context.image_path.name}"):
            # Build ordered tags and get breakdown
            tags, breakdown = self._build_ordered_tags(context)

            if not tags:
                log.debug(f"Section {self.section} is empty - skipping")
                return context

            # Format the output based on section
            if self.section == 2:
                # Section 2 is NL caption - raw text, delimiter ignored
                if len(tags) == 1:
                    output = tags[0]
                else:
                    # Multiple NL entries - join with newlines
                    output = "\n".join(tags)
            else:
                # Sections 0 and 1 are tags - delimited
                output = self._format_tags(tags)

            # Log breakdown
            self._log_breakdown(breakdown)

            # Save to disk
            output_path = self._save_output(context, output)

            # Log output with truncation
            log_truncated("Written", output, max_len=64, level="info", continuation_level="debug")

            # Store result
            result = context.copy()
            result.metadata[f"section_{self.section}_output"] = output
            result.metadata[f"section_{self.section}_path"] = str(output_path)

            return result

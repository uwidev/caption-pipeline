"""
FormatSectionStep: Output only a specific section from the context.
"""

from pathlib import Path

from caption_pipeline.core.context import ImageContext
from caption_pipeline.core.help import step_help
from caption_pipeline.steps.format_base import BaseFormatStep
from caption_pipeline.utils.logging_utils import log


@step_help(
    name="format:section",
    description="Output only a specific section from the context.",
    long_description="""This step outputs a single section from the context to a file.

Sections:
- Section 0: Prepended tags (delimited by delimiter)
- Section 1: Main tags (delimited by delimiter)
- Section 2: Natural language caption (raw text, delimiter ignored)

For section 1 (main tags), the order follows the original script:
Rating → Character tags → General tags

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
            tags = context.get_tags(self.section)

            if not tags:
                log.debug(f"Section {self.section} is empty - skipping")
                return context

            # Apply ordering for section 1 (main tags)
            if self.section == 1:
                tags = self._order_tags(tags, context)

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

            # Save to disk
            output_path = self._save_output(context, output)

            # Log results
            self._log_output(context, output, output_path)

            # Store result
            result = context.copy()
            result.metadata[f"section_{self.section}_output"] = output
            result.metadata[f"section_{self.section}_path"] = str(output_path)

            return result

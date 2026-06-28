"""
FormatSectionStep: Output only a specific section from the context.
"""

from pathlib import Path

from caption_pipeline.core.context import ImageContext
from caption_pipeline.core.help import step_help
from caption_pipeline.core.step import PipelineStep
from caption_pipeline.utils.logging_utils import log


@step_help(
    name="format:section",
    description="Output only a specific section from the context.",
    long_description="""This step outputs a single section from the context to a file.

Sections:
- Section 0: Prepended tags (delimited by delimiter)
- Section 1: Main tags (delimited by delimiter)
- Section 2: Natural language caption (raw text, delimiter ignored)

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
class FormatSectionStep(PipelineStep):
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
        """
        Initialize the format section step.

        Args:
            section: Section to output (0=prepended, 1=main, 2=NL)
            output_dir: Output directory for the file
            suffix: Suffix to add to the output filename
            delimiter: Delimiter for tags (sections 0 and 1)
            use_spaces: Convert underscores to spaces in tags
        """
        self.section: int = section
        self.output_dir: Path = output_dir or Path("./done/")
        self.suffix: str = suffix
        self.delimiter: str = delimiter
        self.use_spaces: bool = use_spaces

    def name(self) -> str:
        return "format:section"

    def validate(self, context: ImageContext) -> bool:
        """Run if the section has content."""
        tags = context.get_tags(self.section)
        return bool(tags)

    def process(self, context: ImageContext) -> ImageContext | None:
        """Output the specified section."""
        with log.section(f"Processing: {context.image_path.name}"):
            tags = context.get_tags(self.section)
            
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
                if self.use_spaces:
                    formatted_tags = [tag.replace("_", " ") for tag in tags]
                else:
                    formatted_tags = tags
                output = self.delimiter.join(formatted_tags)

            # Save to disk
            self.output_dir.mkdir(parents=True, exist_ok=True)
            output_path = self.output_dir / f"{context.image_path.stem}{self.suffix}.txt"
            output_path.write_text(output)
            
            log.info(f"Section {self.section} -> {output_path.name}")
            if len(output) > 100:
                log.debug(f"Output preview: {output[:100]}...")
            else:
                log.debug(f"Output: {output}")

            # Store result
            result = context.copy()
            result.metadata[f"section_{self.section}_output"] = output
            result.metadata[f"section_{self.section}_path"] = str(output_path)

            return result

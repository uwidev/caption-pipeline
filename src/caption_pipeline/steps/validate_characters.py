"""
CharacterValidationStep: Validate that user-provided grounding tags contain valid character references.
"""

from pathlib import Path

from caption_pipeline.core.context import ImageContext
from caption_pipeline.core.help import step_help
from caption_pipeline.core.step import PipelineStep
from caption_pipeline.utils.logging_utils import (
    log,
    section,
)
from caption_pipeline.utils.tag_db import load_character_tags_only


@step_help(
    name="debug:validate_characters",
    description="Validate that user-provided grounding tags contain valid character references.",
    long_description="""This step checks each image's grounding tags to ensure there is a valid
character tag, or 'original'/'borrowed character' is present.

If validation fails, the filename is written to a file for review.

This is useful for:
- Finding images that need character tags added
- Identifying missing character database entries
- Ensuring proper grounding before NL generation

Note: 'original' and 'borrowed character' are treated as intentional user overrides
and are considered valid.""",
    options=[
        {
            "flag": "--output-file PATH",
            "help": "Output file for missing characters",
            "default": "./missing_characters.txt",
        },
    ],
    example="debug:characters --output-file ./missing.txt",
)
class CharacterValidationStep(PipelineStep):
    """
    Validate that user-provided grounding tags contain valid character references.
    """

    SPECIAL_TAGS = {"original", "borrowed_character"}

    def __init__(
        self,
        output_file: Path | str = "./missing_characters.txt",
    ) -> None:
        """
        Initialize the character validation step.

        Args:
            output_file: Path to the output file for missing characters
        """
        self.output_file = Path(output_file)
        self._missing: list[str] = []
        self._character_tag_set: set[str] | None = None

    def name(self) -> str:
        """Return the step's unique identifier."""
        return "debug:characters"

    def validate(self, context: ImageContext) -> bool:
        """Always run (no pre-validation needed)."""
        return True

    def _get_character_tag_set(self) -> set[str]:
        """Lazy load the character tag set."""
        if self._character_tag_set is None:
            self._character_tag_set = load_character_tags_only()
        return self._character_tag_set

    def _is_valid_character_context(self, context: ImageContext) -> tuple[bool, str | None]:
        """
        Check if the context has valid character references.

        Returns:
            Tuple of (is_valid, reason)
        """
        # Check 1: Context has character tags from hints or AI
        if context.has_characters():
            return True, f"character tags: {', '.join(context.character_tags)}"

        # Check 2: Unnamed/original character (user used 'original' or 'borrowed_character')
        if context.has_unnamed_character():
            return True, "unnamed/original character (user override)"

        # Check 3: Check tags directly (in case character tags weren't extracted)
        all_tags = context.get_tags(section=0) + context.get_tags(section=1)
        character_tag_set = self._get_character_tag_set()

        for tag in all_tags:
            # Check if it's a special tag
            if tag in self.SPECIAL_TAGS:
                return True, f"special tag: '{tag}'"

            # Check if it's in the character database
            if tag in character_tag_set:
                return True, f"character tag in database: '{tag}'"

        # No valid character reference found
        return False, "No character tag or user override found"

    def process(self, context: ImageContext) -> ImageContext | None:
        """
        Process a single context by validating its character tags.
        """
        with section(f"Processing: {context.image_path.name}"):
            is_valid, reason = self._is_valid_character_context(context)

            if not is_valid:
                log.warning(f"Missing character validation: {context.image_path.name} - {reason}")
                if context.image_path.name not in self._missing:
                    self._missing.append(context.image_path.name)
            else:
                if context.has_characters():
                    log.debug(
                        f"✓ Valid character(s) found for {context.image_path.name}: {', '.join(context.character_tags)}"
                    )
                elif context.has_unnamed_character():
                    log.debug(f"✓ Unnamed/original character for {context.image_path.name}")
                else:
                    log.debug(
                        f"✓ Character reference found for {context.image_path.name}: {reason}"
                    )

            return context

    def process_batch(self, contexts: list[ImageContext]) -> list[ImageContext]:
        """
        Process all contexts and write results to file.
        """
        if not contexts:
            return contexts

        # Reset missing list
        self._missing = []

        # Process each context
        results: list[ImageContext] = []
        for context in contexts:
            result = self.process(context)
            if result is not None:
                results.append(result)

        # Write missing characters to file
        if self._missing:
            self.output_file.parent.mkdir(parents=True, exist_ok=True)
            with self.output_file.open("w") as f:
                for filename in sorted(self._missing):
                    f.write(f"{Path(filename).stem}\n")
            log.info(
                f"Found {len(self._missing)} images missing character validation. Written to {self.output_file}"
            )
        else:
            # Remove the file if it exists and is empty
            if self.output_file.exists():
                self.output_file.unlink()
                log.debug(f"Removed empty output file: {self.output_file}")
            log.info("All images passed character validation")

        return results

    def write_missing_files(self) -> None:
        """Manually write the missing characters list to file."""
        if self._missing:
            self.output_file.parent.mkdir(parents=True, exist_ok=True)
            with self.output_file.open("w") as f:
                for filename in sorted(self._missing):
                    f.write(f"{filename}\n")
        else:
            if self.output_file.exists():
                self.output_file.unlink()

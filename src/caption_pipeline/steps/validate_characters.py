"""
CharacterValidationStep: Validate that user-provided grounding tags contain valid character references.
"""

from pathlib import Path

from loguru import logger

from caption_pipeline.core.context import ImageContext
from caption_pipeline.core.help import step_help
from caption_pipeline.core.step import PipelineStep
from caption_pipeline.utils.character_extractor import (
    CharacterExtractor,
    get_character_database,
)


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
        {"flag": "--output-file PATH", "help": "Output file for missing characters", "default": "./missing_characters.txt"},
    ],
    example="debug:characters --output-file ./missing.txt",
)
class CharacterValidationStep(PipelineStep):
    """
    Validate that user-provided grounding tags contain valid character references.
    """

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
        self._extractor = CharacterExtractor()
        self._missing: list[str] = []

    def name(self) -> str:
        """Return the step's unique identifier."""
        return "debug:characters"

    def validate(self, context: ImageContext) -> bool:
        """Always run (no pre-validation needed)."""
        return True

    def process(self, context: ImageContext) -> ImageContext | None:
        """
        Process a single context by validating its character tags.
        """
        logger.debug(f"Processing: {context.image_path.name}")

        # Get tags from section 1 (main tags)
        tags = context.get_tags(section=1)
        # Also check section 0 (prepended tags)
        prepended = context.get_tags(section=0)
        all_tags = tags + prepended

        # Check for existing character entries (already extracted at load time)
        has_character_entry = bool(context.character_entries)

        # Check for special tags (user overrides)
        has_original = "original" in all_tags or "original" in tags
        has_borrowed = "borrowed character" in all_tags or "borrowed character" in tags

        # Check for character tags in the tags themselves (in case they weren't extracted)
        has_character_tag = False
        for tag in all_tags:
            if self._extractor.is_character_tag(tag):
                has_character_tag = True
                break

        # Determine if validation passes
        is_valid = False
        reason = None

        if has_character_entry or has_character_tag:
            is_valid = True
        elif has_original:
            is_valid = True
            reason = "'original' user override"
        elif has_borrowed:
            is_valid = True
            reason = "'borrowed character' user override"
        else:
            # Check if any tag exists in the database (last chance)
            db = get_character_database()
            for tag in all_tags:
                normalized = tag.lower().replace(" ", "_").strip("_ ")
                if normalized in db:
                    is_valid = True
                    break
            
            if not is_valid:
                reason = "No character tag or user override found"

        if not is_valid:
            logger.warning(f"Missing character validation: {context.image_path.name} - {reason}")
            if context.image_path.name not in self._missing:
                self._missing.append(context.image_path.name)
        else:
            if has_character_entry:
                char_names = [e.tag for e in context.character_entries]
                logger.debug(f"✓ Valid character(s) found for {context.image_path.name}: {', '.join(char_names)}")
            elif has_original:
                logger.debug(f"✓ 'original' user override for {context.image_path.name}")
            elif has_borrowed:
                logger.debug(f"✓ 'borrowed character' user override for {context.image_path.name}")
            elif has_character_tag:
                logger.debug(f"✓ Character tag found for {context.image_path.name}")

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
            logger.info(f"Found {len(self._missing)} images missing character validation. Written to {self.output_file}")
        else:
            # Remove the file if it exists and is empty
            if self.output_file.exists():
                self.output_file.unlink()
                logger.debug(f"Removed empty output file: {self.output_file}")
            logger.info("All images passed character validation")

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

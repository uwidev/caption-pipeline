F"""
Command-line interface for the caption pipeline.
"""

import argparse
import mimetypes
import shlex
import sys
from pathlib import Path

from loguru import logger

from caption_pipeline.core import PipelineStep, format_step_help, get_step_help
from caption_pipeline.core.context import ImageContext
from caption_pipeline.core.pipeline import Pipeline
from caption_pipeline.steps.debug import DebugStep
from caption_pipeline.steps.fix_danbooru import FixDanbooruStep
from caption_pipeline.steps.fix_overlap import FixOverlapStep
from caption_pipeline.steps.fix_counts import FixCountsStep
from caption_pipeline.steps.format_join import FormatJoinStep
from caption_pipeline.steps.format_section import FormatSectionStep
from caption_pipeline.steps.tag_generate import TagGenerationStep
from caption_pipeline.steps.tag_manipulate import TagManipulateStep
from caption_pipeline.steps.tag_natural_language import TagNaturalLanguageStep
from caption_pipeline.steps.tag_natural_language_filter import TagNaturalLanguageFilterStep
from caption_pipeline.steps.tag_resolve import TagResolveStep
from caption_pipeline.steps.validate_characters import CharacterValidationStep
from caption_pipeline.utils import load_tag_databases
from caption_pipeline.utils.logging_utils import configure_logging, log, log_truncated, section

# Image MIME types supported
SUPPORTED_IMAGE_MIMES: set[str] = {
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/gif",
    "image/bmp",
    "image/tiff",
    "image/avif",
    "image/heic",
    "image/heif",
}

# Cache for character database
_CHARACTER_TAGS: set[str] | None = None

RATING_TAGS = {"safe", "questionable", "explicit", "general", "sensitive"}


def get_character_tags() -> set[str]:
    """Get the set of character tags from the database (cached)."""
    global _CHARACTER_TAGS
    if _CHARACTER_TAGS is None:
        _, characters = load_tag_databases()
        _CHARACTER_TAGS = set(characters)
    return _CHARACTER_TAGS


def normalize_character_tag(tag: str) -> str:
    """
    Normalize a character tag to database format.

    Database format: lowercase_with_underscores

    Input: "albedo (overlord)" or "albedo_(overlord)" or "character:albedo"
    Output: "albedo_(overlord)"# (lowercase + spaces → underscores)
    """
    if not tag:
        return ""

    # Remove "character:" prefix
    if tag.startswith("character:"):
        tag = tag[10:]

    # Convert to lowercase
    tag = tag.lower()

    # Convert spaces to underscores
    tag = tag.replace(" ", "_")

    return tag.strip("_ ")


def setup_logging(debug: bool = False) -> None:
    """Setup logging configuration with colors."""
    configure_logging(debug)


def is_image_file(file_path: Path) -> bool:
    """
    Check if a file is an image using MIME type detection.

    Args:
        file_path: Path to the file

    Returns:
        True if the file is a supported image
    """
    if not file_path.exists() or not file_path.is_file():
        return False

    # Try to detect MIME type from file extension first
    mime_type, _ = mimetypes.guess_type(str(file_path))
    if mime_type and mime_type in SUPPORTED_IMAGE_MIMES:
        return True

    # Fallback: check common image extensions
    # Some systems may not have all MIME types registered
    ext = file_path.suffix.lower()
    if ext in {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".gif",
        ".bmp",
        ".tiff",
        ".tif",
        ".avif",
        ".heic",
        ".heif",
    }:
        return True

    # Try to read file signature for more reliable detection
    try:
        import magic

        # Use python-magic for MIME detection
        mime = magic.from_file(str(file_path), mime=True)
        return mime in SUPPORTED_IMAGE_MIMES
    except ImportError:
        # python-magic not available, fall back to extension detection
        pass
    except Exception as e:
        log.debug(f"Failed to detect MIME type for {file_path}: {e}")

    return False


def find_images_in_directory(
    directory: Path,
    recursive: bool = False,
) -> list[Path]:
    """
    Find all image files in a directory using MIME type detection.

    Args:
        directory: Directory to search
        recursive: Whether to search subdirectories
        supported_mimes: Set of supported MIME types (uses default if None)

    Returns:
        List of image file paths
    """
    if not directory.exists():
        log.error(f"Directory not found: {directory}")
        return []

    if not directory.is_dir():
        # If it's a file, check if it's an image
        if is_image_file(directory):
            return [directory]
        log.warning(f"Not a directory or image file: {directory}")
        return []

    log.info(f"Scanning directory: {directory}")

    image_files: list[Path] = []

    # Walk the directory
    if recursive:
        iterator = directory.rglob("*")
    else:
        iterator = directory.glob("*")

    for file_path in iterator:
        if file_path.is_file() and is_image_file(file_path):
            image_files.append(file_path)

    log.info(f"Found {len(image_files)} image files")

    return image_files


def load_existing_caption(image_path: Path) -> list[list[str]]:
    """
    Load existing caption file if it exists.

    Format: "section0 ||| section1 ||| section2"
    - section0: Prepended tags (comma-separated)
    - section1: Main tags (comma-separated)
    - section2: Natural language caption (SINGLE STRING, NOT split)

    Human-readable tags uses spaces. Convert to underscores to be
    standardized to danbooru datasets.

    Args:
        image_path: Path to the image file

    Returns:
        List of 3 sections: [[prepended_tags], [main_tags], [nl_caption]]
        - Prepended and main tags are lists of strings
        - NL caption is a list with a SINGLE string
    """
    caption_path = image_path.with_suffix(".txt")
    if not caption_path.exists():
        return [[], [], []]

    content = caption_path.read_text().strip()
    if not content:
        return [[], [], []]

    def split_and_underscore(text: str) -> list[str]:
        """Split tags by comma, handling both ', ' and ',', and convert space to underscore."""
        if not text or text.strip() == "":
            return []
        # Split by comma and clean each tag
        tags = [t.strip().replace(" ", "_") for t in text.split(",") if t.strip()]
        return tags

    # Check if we have sections separated by " ||| "
    # Handle case where content might start with "|||" or " |||"
    normalized_content = content

    # If the content starts with "|||" (with or without space), treat first section as empty
    if normalized_content.startswith("|||"):
        normalized_content = " " + normalized_content  # Add space to make parsing consistent
    elif normalized_content.startswith(" |||"):
        # Already has leading space, keep as is
        pass

    if " ||| " in normalized_content:
        sections = normalized_content.split(" ||| ")

        # Strip whitespace from each section
        sections = [s.strip() for s in sections]

        # Handle the case where the first section is empty (content starts with "|||")
        if sections and sections[0] == "":
            sections = sections[1:]  # Remove the empty first section
            # Now we have [section0, section1, section2] but section0 was empty
            # So we need to add an empty section0 back
            sections = [""] + sections

        parsed_sections = []

        for idx, section_text in enumerate(sections):
            if idx == 2:  # Section 2 is NL - KEEP AS SINGLE STRING
                # Store the ENTIRE section as a single string
                parsed_sections.append([section_text.strip()])
            else:
                # Sections 0 and 1 are tags - split by commas
                parsed_sections.append(split_and_underscore(section_text))

        # Ensure we have exactly 3 sections
        while len(parsed_sections) < 3:
            parsed_sections.append([])

        return parsed_sections
    else:
        # No sections - treat as single tag list (section 1)
        tags = split_and_underscore(content)
        return [[], tags, []]


def extract_rating(tags: list[str]) -> tuple[list[str], str | None]:
    """
    Validate that only one rating tag exists in the tags.

    Args:
        tags: List of tags to check

    Returns:
        Tuple of (filtered_tags, rating)
        - filtered_tags: Tags without rating tags
        - rating: The rating tag if found, None otherwise

    Raises:
        ValueError: If multiple rating tags are found
    """
    found_ratings = []
    filtered_tags = []

    for tag in tags:
        normalized = tag.lower().strip()
        if normalized in RATING_TAGS:
            found_ratings.append(tag)
        else:
            filtered_tags.append(tag)

    if len(found_ratings) > 1:
        log.warning(
            f"Multiple rating tags found: {', '.join(found_ratings)}. "
            f"Using '{found_ratings[0]}' as the rating."
        )
        # Keep the first rating tag found, remove others
        rating = found_ratings[0]
    elif len(found_ratings) == 1:
        rating = found_ratings[0]
    else:
        rating = None

    return filtered_tags, rating


def extract_character_hints(tags: list[str]) -> tuple[list[str], list[str]]:
    """
    Extract character tags from user hints using the central tag database.

    Rules:
    1. Tags with "character:" prefix are ALWAYS characters (user explicitly said so)
    2. No prefix? Cross-check with tag database to find character tags

    Args:
        tags: List of tags to process

    Returns:
        Tuple of (remaining_tags, character_tags)
    """

    characters: list[str] = []
    remaining: list[str] = []
    explicit_hints: list[str] = []

    # Try 1: Look for character: prefixed tags first
    for tag in tags:
        if tag.startswith("character:"):
            # Extract the actual name
            char_name = tag[10:].strip().lower().replace(" ", "_")
            if char_name:
                explicit_hints.append(char_name)

    if explicit_hints:
        # We found explicit characters earlier, keep everything except character tags
        remaining.extend([tag for tag in tags if not tag.startswith("character:")])
        characters = explicit_hints
        log.debug(f"Found {len(characters)} explicit character hints")
        return remaining, characters

    # Try 2: Check each tag if they are a character, just not prefixed
    from caption_pipeline.utils.tag_db import load_character_tags_only

    character_tag_set = load_character_tags_only()
    tags_to_remove = []

    for tag in tags:
        if tag in character_tag_set:
            characters.append(tag)
            tags_to_remove.append(tag)
            log.debug(f"Found character in tag database: '{tag}")
        else:
            remaining.append(tag)

    if characters:
        return remaining, characters

    return remaining, characters


def get_all_step_classes() -> list[type]:
    """Get all step classes with help metadata."""
    return [
        TagGenerationStep,
        TagResolveStep,
        TagManipulateStep,
        TagNaturalLanguageStep,
        TagNaturalLanguageFilterStep,
        FormatJoinStep,
        FormatSectionStep,
        CharacterValidationStep,
        FixOverlapStep,
        FixCountsStep,
        FixDanbooruStep,
        DebugStep,
    ]


def parse_steps(args: argparse.Namespace) -> list[PipelineStep]:
    """Parse command-line steps into pipeline steps."""
    steps: list[PipelineStep] = []

    for step_str in args.steps:
        parts = shlex.split(step_str)
        step_name = parts[0]

        match step_name:
            case "debug:validate_characters" | "debug:characters" | "debug:char":
                output_file = "./missing_characters.txt"

                i = 1
                while i < len(parts):
                    match parts[i]:
                        case "--output-file":
                            output_file = Path(parts[i + 1])
                            i += 2
                        case _:
                            raise ValueError(
                                f"Unknown flag '{parts[i]}' for step '{step_name}'. "
                                f"Available flags: --output-file"
                            )

                steps.append(
                    CharacterValidationStep(
                        output_file=output_file,
                    )
                )

            case "fix:danbooru_only" | "fix:danbooru" | "fix:db":
                whitelist = []
                target_section = 1  # Renamed from 'section'

                i = 1
                while i < len(parts):
                    match parts[i]:
                        case "--whitelist":
                            whitelist = parts[i + 1].split(",")
                            i += 2
                        case "--section":
                            target_section = int(parts[i + 1])
                            i += 2
                        case _:
                            raise ValueError(
                                f"Unknown flag '{parts[i]}' for step '{step_name}'. "
                                f"Available flags: --whitelist, --section"
                            )

                steps.append(
                    FixDanbooruStep(
                        whitelist=whitelist,
                        target_section=target_section,
                    )
                )

            case "fix:overlap" | "fix:drop":
                target_section = -1
                keep_scored = False
                keep_hints = False

                i = 1
                while i < len(parts):
                    match parts[i]:
                        case "--section":
                            target_section = int(parts[i + 1])
                            i += 2
                        case "--keep-scored":
                            keep_scored = True
                            i += 1
                        case "--keep-hints":
                            keep_hints = True
                            i += 1
                        case _:
                            raise ValueError(
                                f"Unknown flag '{parts[i]}' for step '{step_name}'. "
                                f"Available flags: --section, --keep-scored, --keep-hints"
                            )

                steps.append(
                    FixOverlapStep(
                        target_section=target_section,
                        keep_scored=keep_scored,
                        keep_hints=keep_hints,
                    )
                )

            case "fix:counts" | "fix:cnt":
                target_section = 1

                i = 1
                while i < len(parts):
                    match parts[i]:
                        case "--section":
                            target_section = int(parts[i + 1])
                            i += 2
                        case _:
                            raise ValueError(
                                f"Unknown flag '{parts[i]}' for step '{step_name}'. "
                                f"Available flags: --section"
                            )

                steps.append(
                    FixCountsStep(
                        target_section=target_section,
                    )
                )

            case "tag:generate" | "tag:gen":
                threshold = 0.35
                character_threshold = 0.75
                whitelist = []
                blacklist = []
                infer_characters = False
                unload_models = True
                use_hints = True

                i = 1
                while i < len(parts):
                    match parts[i]:
                        case "--threshold" | "--thresh":
                            threshold = float(parts[i + 1])
                            i += 2
                        case "--character-threshold" | "--cthresh":
                            character_threshold = float(parts[i + 1])
                        case "--whitelist":
                            whitelist = parts[i + 1].split(",")
                            i += 2
                        case "--blacklist":
                            blacklist = parts[i + 1].split(",")
                            i += 2
                        case "--infer-characters":
                            infer_characters = True
                            i += 1
                        case "--no-unload-models":
                            unload_models = False
                            i += 1
                        case "--use-hints":
                            use_hints = True
                            i += 1
                        case "--no-use-hints":
                            use_hints = False
                            i += 1
                        case _:
                            raise ValueError(
                                f"Unknown flag '{parts[i]}' for step '{step_name}'. "
                                f"Available flags: --threshold, --thresh, --character-threshold "
                                f"--cthresh --whitelist, --blacklist --infer-characters --no-unload-models "
                                f"--use-hints, --no-use-hints"
                            )

                steps.append(
                    TagGenerationStep(
                        threshold=threshold,
                        character_threshold=character_threshold,
                        whitelist=whitelist,
                        blacklist=blacklist,
                        infer_characters=infer_characters,
                        unload_models_after_batch=unload_models,
                        use_user_hints=use_hints,
                    )
                )

            case "tag:resolve" | "tag:fix":
                mode = "drop"
                max_padding = 30
                max_windows = 0
                force_windows = 0
                threshold = None
                max_tags = 0
                keep_hints = True

                i = 1
                while i < len(parts):
                    match parts[i]:
                        case "--mode":
                            mode = parts[i + 1]
                            i += 2
                        case "--max-padding":
                            max_padding = int(parts[i + 1])
                            i += 2
                        case "--max-windows":
                            max_windows = int(parts[i + 1])
                            i += 2
                        case "--force-windows":
                            force_windows = int(parts[i + 1])
                            i += 2
                        case "--threshold":
                            threshold = float(parts[i + 1])
                            i += 2
                        case "--max-tags":
                            max_tags = int(parts[i + 1])
                            i += 2
                        case "--no-keep-hints":
                            keep_hints = False
                            i += 1
                        case _:
                            raise ValueError(
                                f"Unknown flag '{parts[i]}' for step '{step_name}'. "
                                f"Available flags: --mode, --max-padding, --max-windows, "
                                f"--force-windows, --threshold, --max-tags, --no-keep-hints"
                            )

                steps.append(
                    TagResolveStep(
                        mode=mode,
                        max_padding=max_padding,
                        max_windows=max_windows,
                        force_windows=force_windows,
                        threshold=threshold,
                        max_tags=max_tags,
                        keep_hints=keep_hints,
                    )
                )

            case "tag:manipulate" | "tag:do":
                operation = "prepend"
                tags = []
                target_section = 1
                remove_duplicates = True
                target_position = -1

                i = 1
                while i < len(parts):
                    match parts[i]:
                        case "--operation" | "--op" | "--mode":
                            operation = parts[i + 1]
                            i += 2
                        case "--tags":
                            tags_str = parts[i + 1]
                            if "," in tags_str:
                                tags = [t.strip() for t in tags_str.split(",")]
                            else:
                                tags = [tags_str]
                            i += 2
                        case "--section" | "--on":
                            target_section = int(parts[i + 1])
                            i += 2
                        case "--no-remove-duplicates":
                            remove_duplicates = False
                            i += 1
                        case "--target-position":
                            target_position = int(parts[i + 1])
                            i += 2
                        case _:
                            raise ValueError(
                                f"Unknown flag '{parts[i]}' for step '{step_name}'. "
                                f"Available flags: --operation, --op, --mode, --tags, --section, --on "
                                f"--no-remove-duplicates, --target-position"
                            )

                if tags:
                    steps.append(
                        TagManipulateStep(
                            operation=operation,
                            tags=tags,
                            target_section=target_section,
                            remove_duplicates=remove_duplicates,
                            target_position=target_position,
                        )
                    )

            case "tag:natural_language" | "tag:nl":
                # Natural language captioning with ToriiGate
                caption_type = "short"
                model_path = None
                mmproj_path = None
                server_port = 8081
                server_host = "127.0.0.1"
                server_log_file = None
                auto_manage_server = True
                debug = args.debug

                i = 1
                while i < len(parts):
                    match parts[i]:
                        case "--type":
                            caption_type = parts[i + 1]
                            i += 2
                        case "--model-path":
                            model_path = Path(parts[i + 1])
                            i += 2
                        case "--mmproj-path":
                            mmproj_path = Path(parts[i + 1])
                            i += 2
                        case "--port":
                            server_port = int(parts[i + 1])
                            i += 2
                        case "--host":
                            server_host = parts[i + 1]
                            i += 2
                        case "--log-file":
                            server_log_file = Path(parts[i + 1])
                            i += 2
                        case "--no-auto-server":
                            auto_manage_server = False
                            i += 1
                        case _:
                            raise ValueError(
                                f"Unknown flag '{parts[i]}' for step '{step_name}'. "
                                f"Available flags: --type, --model-path, --mmproj-path, "
                                f"--port, --host, --log-file, --no-auto-server"
                            )

                steps.append(
                    TagNaturalLanguageStep(
                        caption_type=caption_type,
                        model_path=model_path,
                        mmproj_path=mmproj_path,
                        server_port=server_port,
                        server_host=server_host,
                        server_log_file=server_log_file,
                        auto_manage_server=auto_manage_server,
                        debug=debug,
                    )
                )

            case "tag:tag_natural_language_filter" | "tag:nl_filter" | "tag:nlf":
                # Filter natural language captions through Ollama
                model = "deepseek-r1:14b"
                ollama_url = "http://localhost:11434/api/chat"
                temperature = 0.3
                max_retries = 3
                timeout = 120
                backup = True
                keep_alive = 3600

                i = 1
                while i < len(parts):
                    match parts[i]:
                        case "--model":
                            model = parts[i + 1]
                            i += 2
                        case "--url":
                            ollama_url = parts[i + 1]
                            i += 2
                        case "--temperature":
                            temperature = float(parts[i + 1])
                            i += 2
                        case "--retries":
                            max_retries = int(parts[i + 1])
                            i += 2
                        case "--timeout":
                            timeout = int(parts[i + 1])
                            i += 2
                        case "--no-backup":
                            backup = False
                            i += 1
                        case "--keep-alive":
                            keep_alive = int(parts[i + 1])
                            i += 2
                        case _:
                            raise ValueError(
                                f"Unknown flag '{parts[i]}' for step '{step_name}'. "
                                f"Available flags: --model, --url, --temperature, --retries, "
                                f"--timeout, --no-backup, --keep-alive"
                            )

                steps.append(
                    TagNaturalLanguageFilterStep(
                        model=model,
                        ollama_url=ollama_url,
                        temperature=temperature,
                        max_retries=max_retries,
                        timeout=timeout,
                        backup_original=backup,
                        keep_alive=keep_alive,
                    )
                )

            case "format:join" | "format:j":
                delimiter = " ||| "
                output_dir = Path("./done/")
                tag_suffix = ""
                deduplicate = True
                use_spaces = True

                i = 1
                while i < len(parts):
                    match parts[i]:
                        case "--delimiter":
                            delimiter = parts[i + 1]
                            i += 2
                        case "--output-dir":
                            output_dir = Path(parts[i + 1])
                            i += 2
                        case "--tag-suffix":
                            tag_suffix = parts[i + 1]
                            i += 2
                        case "--no-deduplicate":
                            deduplicate = False
                            i += 1
                        case "--no-spaces":
                            use_spaces = False
                            i += 1
                        case _:
                            raise ValueError(
                                f"Unknown flag '{parts[i]}' for step '{step_name}'. "
                                f"Available flags: --delimiter, --output-dir, --tag-suffix, "
                                f"--no-deduplicate, --no-spaces"
                            )

                steps.append(
                    FormatJoinStep(
                        delimiter=delimiter,
                        output_dir=output_dir,
                        tag_suffix=tag_suffix,
                        deduplicate_tags=deduplicate,
                        use_spaces=use_spaces,
                    )
                )

            case "format:section" | "format:s":
                target_section = 1
                output_dir = Path("./done/")
                suffix = ""
                delimiter = ", "
                use_spaces = True

                i = 1
                while i < len(parts):
                    match parts[i]:
                        case "--section":
                            target_section = int(parts[i + 1])
                            i += 2
                        case "--output-dir":
                            output_dir = Path(parts[i + 1])
                            i += 2
                        case "--suffix":
                            suffix = parts[i + 1]
                            i += 2
                        case "--delimiter":
                            delimiter = parts[i+1]
                            i += 2
                        case "--no-spaces":
                            use_spaces = False
                            i += 1
                        case _:
                            raise ValueError(
                                f"Unknown flag '{parts[i]}' for step '{step_name}'. "
                                f"Available flags: --section, --output-dir, --suffix, --delimiter, --no-spaces"
                            )

                steps.append(
                    FormatSectionStep(
                        target_section=target_section,
                        output_dir=output_dir,
                        suffix=suffix,
                        delimiter=delimiter,
                        use_spaces=use_spaces,
                    )
                )

            case "debug":
                steps.append(DebugStep())

            case _:
                raise ValueError(f"Unknown step: {step_name}")

    return steps


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Caption Pipeline - Modular image captioning pipeline for diffusion model training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate tags and NL captions for all images
  caption-pipeline process --input ./img/ --steps "tag:generate --threshold 0.35" --steps "tag:nl" --steps "format:join"

  # NL captioning only with custom server settings
  caption-pipeline process --input ./img/ --steps "tag:nl --force --server-port 8082 --server-log-file ./server.log" --steps "format:join"

  # Recursive directory scanning
  caption-pipeline process --input ./img/ --recursive --steps "tag:nl --no-require-tags" --steps "format:join"

Use --help-steps to see detailed step reference.
""",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Process command
    process_parser = subparsers.add_parser(
        "process",
        help="Run the caption pipeline",
    )
    process_parser.add_argument(
        "--input",
        required=True,
        help="Input file or directory (e.g., ./img/ or ./img/image.webp)",
    )
    process_parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search subdirectories recursively for images",
    )
    process_parser.add_argument(
        "--steps",
        action="append",
        required=True,
        help="Pipeline steps to run (see --help-steps)",
    )
    process_parser.add_argument(
        "--output-dir",
        default="./done/",
        help="Output directory for processed files (default: ./done/)",
    )
    process_parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    # Version command
    version_parser = subparsers.add_parser("version", help="Show version")

    # Add --help-steps
    parser.add_argument(
        "--help-steps",
        action="store_true",
        help="Show detailed help for all pipeline steps",
    )

    args = parser.parse_args()

    if args.help_steps:
        print("=" * 80)
        print("CAPTION PIPELINE - STEP REFERENCE")
        print("=" * 80)
        print("")

        step_classes = get_all_step_classes()
        for cls in step_classes:
            meta = get_step_help(cls)
            if meta:
                print(format_step_help(meta))
                print("")

        sys.exit(0)

    if args.command == "version":
        print("Caption Pipeline v0.1.0")
        return

    if args.command == "process":
        setup_logging(args.debug)

        with section("Starting caption pipeline"):
            log.debug("Debug mode enabled")

            # Find input files
            input_path = Path(args.input)

            if input_path.is_dir():
                log.info(f"Processing directory: {input_path}")
                input_files = find_images_in_directory(
                    input_path,
                    recursive=args.recursive,
                )
            elif input_path.is_file():
                if is_image_file(input_path):
                    input_files = [input_path]
                else:
                    log.error(f"File is not a supported image: {input_path}")
                    return
            else:
                log.error(f"Input path does not exist: {input_path}")
                return

            if not input_files:
                log.warning(f"No image files found in {input_path}")
                return

            log.info(f"Found {len(input_files)} image files to process")

            pipeline = Pipeline(error_handling="skip")
            steps = parse_steps(args)
            for step in steps:
                pipeline.add_step(step)

            contexts: list[ImageContext] = []

            with section(f"Loading {len(input_files)} images"):
                for file_path in input_files:
                    with section(f"Processing: {file_path.name}"):
                        # Load the caption tags from existing .txt file
                        tags = load_existing_caption(file_path)

                        # Section 0: Prepended tags
                        if tags[0]:
                            tags_str = ", ".join(tags[0])
                            log_truncated(f"Prepended ({len(tags[0])})", tags_str, max_len=64)
                        else:
                            log.info("Prepended: (none)")

                        # Section 1: Main tags
                        if tags[1]:
                            log.info(
                                f"Main ({len(tags[1])}): {', '.join(tags[1][:10])}{'...' if len(tags[1]) > 10 else ''}"
                            )
                        else:
                            log.info("Main: (none)")

                        # Section 2: NL caption
                        if tags[2] and tags[2][0]:
                            caption_preview = (
                                tags[2][0][:100] + "..." if len(tags[2][0]) > 100 else tags[2][0]
                            )
                            log.info(f"NL: {caption_preview}")
                        else:
                            log.info("NL: (none)")

                        # Combine sections 0 and 1 for processing
                        all_tags = tags[0] + tags[1]

                        # Extract rating FIRST (removes rating tags from all_tags)
                        tags_without_ratings, rating = extract_rating(all_tags)

                        # Log extracted rating at INFO level
                        if rating:
                            log.info(f"Extracted rating: {rating}")
                        else:
                            log.info("Extracted rating: (none)")

                        # Extract character hints from tags without ratings
                        remaining_tags, character_tags = extract_character_hints(tags_without_ratings)

                        # Log extracted characters at INFO level
                        if character_tags:
                            log.info(f"Characters ({len(character_tags)}): {', '.join(character_tags)}")
                        else:
                            log.info("Characters: (none)")

                        # Reconstruct the tag sections
                        modified_tags = [
                            [],  # section 0 - prepended tags
                            remaining_tags,  # section 1 - main tags
                            tags[2] if len(tags) > 2 else [],  # section 2 - NL caption
                        ]

                        # Create the context
                        context = ImageContext(
                            image_path=file_path,
                            source_path=file_path,
                            tags=modified_tags,
                            original_tags=tags,
                            character_tags=character_tags,
                            rating=rating,
                        )
                        contexts.append(context)

            results = pipeline.run(contexts)

            log.info(f"Processed {len(results)} images")

            for context in results:
                context.save_image()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()

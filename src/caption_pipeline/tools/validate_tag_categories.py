"""
Validate tag categories file integrity.

This tool checks a tag categories file for common issues and provides
warnings/errors to help maintain a clean database.
"""

from pathlib import Path

from caption_pipeline.tools.tag_utils import sort_tags_by_object
from caption_pipeline.utils.logging_utils import log

# Known valid categories (from tag_patterns.py and tag_categories.txt)
KNOWN_CATEGORIES = {
    "rating",
    "bodies",
    "character name or series",
    "body parts",
    "wearables",
    "shot composition",
    "pose",
    "action",
    "expressions",
    "effects",
    "atmosphere",
    "environment",
    "uncertain",
}


def normalize_section_name(name: str) -> str:
    """Normalize a section name (trim whitespace, lowercase)."""
    return name.strip().lower()


def normalize_tag(tag: str) -> str:
    """Normalize a tag (lowercase, underscores to spaces, remove extra spaces)."""
    normalized = tag.lower().replace("_", " ")
    return " ".join(normalized.split())


def validate_tag_categories(
    file_path: Path,
    fix: bool = False,
    verbose: bool = False,
) -> bool:
    """
    Validate a tag categories file for integrity issues.

    Args:
        file_path: Path to the tag_categories.txt file.
        fix: Automatically fix issues where possible.
        verbose: Show detailed output.

    Returns:
        True if valid (or fixed), False if unfixable errors exist.

    Validation checks:
        1. File exists and is readable.
        2. Section headers are properly formatted [category].
        3. Section headers are normalized (trimmed, lowercase).
        4. Tags are normalized (lowercase, spaces instead of underscores).
        5. No duplicate tags within a section (auto-fix with --fix).
        6. Warn on duplicate tags across sections (manual resolution needed).
        7. All section names are in the standard category list (warn on unknown).
        8. No empty tags or blank lines in sections.
        9. Tags are sorted alphabetically within sections (auto-fix with --fix).
        10. No trailing whitespace.
    """
    if not file_path.exists():
        log.error(f"File not found: {file_path}")
        return False

    # Read the file
    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception as e:
        log.error(f"Failed to read {file_path}: {e}")
        return False

    lines = content.splitlines()
    data: dict[str, set[str]] = {}
    section_order: list[str] = []
    current_section: str | None = None
    line_num = 0
    errors = 0
    warnings = 0
    fixed = 0

    # First pass: parse and detect issues
    for line in lines:
        line_num += 1
        original_line = line
        stripped = line.strip()

        # Allow blank lines
        if not stripped:
            continue

        # Check for section header
        if stripped.startswith("[") and stripped.endswith("]"):
            raw_section_name = stripped[1:-1].strip()
            if not raw_section_name:
                log.error(f"Line {line_num}: Empty section name")
                errors += 1
                continue

            section_name = normalize_section_name(raw_section_name)

            # Check if section name has extra whitespace
            if raw_section_name != raw_section_name.strip():
                log.warning(
                    f"Line {line_num}: Section name has extra whitespace: '{raw_section_name}'"
                )
                warnings += 1
                if fix:
                    # We'll fix this when writing the file
                    fixed += 1

            # Check if section name is known
            if section_name not in KNOWN_CATEGORIES:
                log.warning(
                    f"Line {line_num}: Unknown section '{section_name}' (not in known categories)"
                )
                warnings += 1

            if section_name in data:
                log.error(f"Line {line_num}: Duplicate section '{section_name}'")
                errors += 1
                continue

            data[section_name] = set()
            section_order.append(section_name)
            current_section = section_name
            continue

        # Tag line
        if current_section is None:
            log.error(f"Line {line_num}: Tag outside any section: '{stripped}'")
            errors += 1
            continue

        # Normalize the tag
        normalized = normalize_tag(stripped)

        if not normalized:
            log.error(f"Line {line_num}: Empty tag")
            errors += 1
            continue

        # Check if tag has underscores (should be spaces)
        if "_" in stripped:
            log.warning(
                f"Line {line_num}: Tag uses underscores: '{stripped}' -> should be '{normalized}'"
            )
            warnings += 1

        # Check for trailing whitespace
        if original_line != original_line.rstrip():
            log.warning(f"Line {line_num}: Trailing whitespace")
            warnings += 1
            if fix:
                fixed += 1

        # Check if tag is already in this section (duplicate)
        if normalized in data[current_section]:
            log.warning(
                f"Line {line_num}: Duplicate tag '{normalized}' in section '{current_section}'"
            )
            warnings += 1
            if fix:
                # We'll deduplicate when writing
                fixed += 1
            continue

        data[current_section].add(normalized)

    # Second pass: check for duplicates across sections
    tag_to_sections: dict[str, list[str]] = {}
    for section, tags in data.items():
        for tag in tags:
            tag_to_sections.setdefault(tag, []).append(section)

    duplicate_tags = {
        tag: sections for tag, sections in tag_to_sections.items() if len(sections) > 1
    }
    if duplicate_tags:
        log.warning(f"Found {len(duplicate_tags)} tags that appear in multiple sections:")
        for tag, sections in sorted(duplicate_tags.items()):
            log.warning(f"  '{tag}' appears in: {', '.join(sections)}")
        warnings += len(duplicate_tags)
        log.warning("Manual resolution required: decide which section each tag should belong to.")

    # Third pass: check section order and sorting
    for section, tags in data.items():
        sorted_tags = sorted(tags)
        if list(tags) != sorted_tags:
            log.warning(f"Section '{section}': Tags are not sorted alphabetically")
            warnings += 1
            if fix:
                data[section] = set(sorted_tags)
                fixed += len(tags)
                if verbose:
                    log.debug(f"  Sorted {len(tags)} tags in section '{section}'")

    # Write fixed file if changes were made
    if fix and (
        fixed > 0 or warnings > 0 or any(normalize_section_name(s) != s for s in section_order)
    ):
        # Normalize section names if needed
        normalized_sections = {}
        for section in list(section_order):
            normalized = normalize_section_name(section)
            if normalized != section:
                log.info(f"Normalizing section name: '{section}' -> '{normalized}'")
                normalized_sections[section] = normalized
                fixed += 1

        # Build new content
        new_content = []
        for idx, section in enumerate(section_order):
            section_key = normalized_sections.get(section, section)
            if section_key not in data:
                continue
            new_content.append(f"[{section_key}]")
            # Use custom sort by object
            for tag in sort_tags_by_object(data[section_key]):
                new_content.append(tag)
            if idx < len(section_order) - 1:
                new_content.append("")  # blank line between sections

        # Backup first
        backup_path = file_path.with_suffix(file_path.suffix + ".bak")
        if backup_path.exists():
            # Find an available backup name
            counter = 2
            while True:
                new_backup = file_path.with_suffix(file_path.suffix + f".bak{counter}")
                if not new_backup.exists():
                    backup_path = new_backup
                    break
                counter += 1

        file_path.rename(backup_path)
        log.info(f"Backed up original to {backup_path}")

        # Write fixed file
        file_path.write_text("\n".join(new_content), encoding="utf-8")
        log.info(f"Fixed {fixed} issues and wrote to {file_path}")

    # Summary
    if errors == 0 and warnings == 0:
        log.info(f"✓ Valid: No issues found in {file_path}")
        return True

    log.info(f"Validation complete: {errors} errors, {warnings} warnings")
    if fix and fixed > 0:
        log.info(f"Fixed {fixed} issues automatically")

    if errors > 0:
        log.error("File has errors that prevent validation")
        return False

    if warnings > 0:
        log.warning("File has warnings that should be addressed")
        return True

    return True

"""
Merge tag categories files.

This tool merges two tag categories files with intelligent conflict resolution.
"""

import shutil
from pathlib import Path

from caption_pipeline.utils.tag_utils import sort_tags_by_object
from caption_pipeline.utils.logging_utils import log


def parse_tag_categories_file(file_path: Path) -> tuple[dict[str, set[str]], list[str]]:
    """
    Parse a tag categories file into (data, section_order).

    data: dict section -> set of tags.
    section_order: list of sections in the order they appear in the file.

    Returns (empty dict, empty list) if file doesn't exist.
    Raises ValueError on format errors.
    """
    if not file_path.exists():
        return {}, []

    data: dict[str, set[str]] = {}
    section_order: list[str] = []
    current_section: str | None = None
    line_num = 0

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line_num += 1
            stripped = line.strip()
            if not stripped:
                continue

            if stripped.startswith("[") and stripped.endswith("]"):
                section_name = stripped[1:-1].strip()
                if not section_name:
                    raise ValueError(f"Empty section name on line {line_num}")
                if section_name in data:
                    raise ValueError(f"Duplicate section '{section_name}' on line {line_num}")
                data[section_name] = set()
                section_order.append(section_name)
                current_section = section_name
            else:
                if current_section is None:
                    raise ValueError(f"Tag outside any section on line {line_num}")
                tag = stripped
                if not tag:
                    raise ValueError(f"Empty tag on line {line_num}")
                data[current_section].add(tag)

    return data, section_order


def write_tag_categories_file(
    file_path: Path, data: dict[str, set[str]], section_order: list[str]
) -> None:
    """
    Write a tag categories dict to a file, preserving section order.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:
        for idx, section in enumerate(section_order):
            if section not in data:
                continue
            f.write(f"[{section}]\n")
            # Use custom sort by object
            for tag in sort_tags_by_object(data[section]):
                f.write(f"{tag}\n")
            if idx < len(section_order) - 1:
                f.write("\n")


def merge_tag_categories(
    target_path: Path,
    source_path: Path,
    output_path: Path | None = None,
    trust_source: bool | None = None,
    backup_suffix: str = ".bak",
    no_backup: bool = False,
) -> bool:
    """
    Merge source file into target file.

    Args:
        target_path: The target (base) file to merge into.
        source_path: The source file to merge from.
        output_path: Path for output (default: overwrite target_path).
        trust_source: If True, source wins in proper conflicts.
                      If False, target wins.
                      If None, abort on proper conflicts.
        backup_suffix: Suffix for backup file (default: .bak).
        no_backup: Skip creating a backup.

    Returns:
        True if successful, False if errors occurred.

    Conflict resolution:
        - If one side has a tag in "uncertain" and the other has it in a proper category,
          the proper category wins (regardless of trust_source).
        - If both sides have proper sections:
          - If trust_source is True, use source's section.
          - If trust_source is False, use target's section.
          - If trust_source is None, abort with a formatted conflict table.
    """
    # Parse target
    target_data, target_order = parse_tag_categories_file(target_path)

    # Parse source
    try:
        source_data, source_order = parse_tag_categories_file(source_path)
    except ValueError as e:
        log.error(f"Invalid source file: {e}")
        return False

    # Validate no duplicate tags across sections in source file
    tag_to_section_source: dict[str, str] = {}
    for section, tags in source_data.items():
        for tag in tags:
            if tag in tag_to_section_source:
                log.error(
                    f"Tag '{tag}' appears in both '{tag_to_section_source[tag]}' and '{section}' in source"
                )
                return False
            tag_to_section_source[tag] = section

    # Build map of where tags exist in target
    tag_to_section_target: dict[str, str] = {}
    for section, tags in target_data.items():
        for tag in tags:
            tag_to_section_target[tag] = section

    # Merge with conflict resolution
    errors = 0
    conflicts: list[dict[str, str]] = []  # list of dicts for proper conflicts when trust is None
    uncertain_sections = {"uncertain"}

    # Start with target as base
    merged_data = target_data.copy()
    merged_order = target_order.copy()

    for section, tags in source_data.items():
        if section not in merged_data:
            merged_data[section] = set()
            merged_order.append(section)

        for tag in tags:
            existing_section = tag_to_section_target.get(tag)

            if existing_section is not None and existing_section != section:
                # Conflict: tag exists in both but different sections
                existing_is_uncertain = existing_section in uncertain_sections
                new_is_uncertain = section in uncertain_sections

                if existing_is_uncertain and not new_is_uncertain:
                    # Target has it in "uncertain" – use source's proper section
                    log.info(
                        f"Tag '{tag}' moved from '{existing_section}' (target, uncertain) "
                        f"to '{section}' (source)"
                    )
                    merged_data[existing_section].discard(tag)
                    merged_data[section].add(tag)
                    tag_to_section_target[tag] = section
                    continue

                elif new_is_uncertain and not existing_is_uncertain:
                    # Source has it in "uncertain" – keep target's proper section
                    log.info(
                        f"Tag '{tag}' remains in '{existing_section}' (target) "
                        f"(ignoring '{section}' from source, uncertain)"
                    )
                    continue

                elif existing_is_uncertain and new_is_uncertain:
                    # Both are uncertain – keep target's version
                    log.debug(f"Tag '{tag}' remains in '{existing_section}' (both uncertain)")
                    continue

                else:
                    # Both are proper – conflict
                    if trust_source is None:
                        conflicts.append(
                            {
                                "tag": tag,
                                "target_section": existing_section,
                                "source_section": section,
                            }
                        )
                        continue
                    elif trust_source:
                        # Trust source – use source's section
                        log.info(
                            f"Tag '{tag}' moved from '{existing_section}' (target) "
                            f"to '{section}' (source) [trusting source]"
                        )
                        merged_data[existing_section].discard(tag)
                        merged_data[section].add(tag)
                        tag_to_section_target[tag] = section
                    else:
                        # Trust target – keep target's section
                        log.info(
                            f"Tag '{tag}' remains in '{existing_section}' (target) "
                            f"(ignoring '{section}' from source) [trusting target]"
                        )
                    continue

            if existing_section == section:
                # Tag already exists in the same section – skip (deduplicated)
                continue

            # No conflict – add the tag
            merged_data[section].add(tag)
            tag_to_section_target[tag] = section

    # If we have conflicts and trust is None, print formatted table and abort
    if conflicts:
        log.error(f"Conflicts found ({len(conflicts)}):")
        # Determine column widths
        tag_width = max(len(c["tag"]) for c in conflicts)
        tag_width = max(tag_width, len("tag"))
        target_width = max(len(c["target_section"]) for c in conflicts)
        target_width = max(target_width, len("target"))
        source_width = max(len(c["source_section"]) for c in conflicts)
        source_width = max(source_width, len("source"))

        # Header
        log.error(f"  {'tag':<{tag_width}}  {'target':<{target_width}}  {'source':<{source_width}}")
        # Rows
        for c in conflicts:
            log.error(
                f"  {c['tag']:<{tag_width}}  {c['target_section']:<{target_width}}  "
                f"{c['source_section']:<{source_width}}"
            )
        log.error("Manually resolve or use --trust to trust ALL file's tag categories.")
        return False

    # Remove any empty sections that were left behind
    empty_sections = [s for s, tags in merged_data.items() if not tags]
    for s in empty_sections:
        del merged_data[s]
        if s in merged_order:
            merged_order.remove(s)

    if errors > 0:
        log.error(f"Aborting due to {errors} unresolvable conflicts")
        return False

    # Backup
    if not no_backup and target_path.exists():
        backup_path = target_path.with_suffix(target_path.suffix + backup_suffix)
        # Avoid clobbering existing backup
        counter = 1
        while backup_path.exists():
            backup_path = target_path.with_suffix(target_path.suffix + f".bak{counter}")
            counter += 1
        shutil.copy2(target_path, backup_path)
        log.info(f"Backed up original to {backup_path}")

    # Write merged
    output_path = output_path or target_path
    write_tag_categories_file(output_path, merged_data, merged_order)
    log.info(f"Merged file written to {output_path}")
    return True

"""
Tag Database Utilities - Centralized loading of all danbooru tag sources.

This module loads and caches tags from all sources:
1. tags_v0.9_13k.json (PixAI) - REQUIRED - contains both general and character tags
2. selected_tags.csv (WD14) - OPTIONAL - categories: 0=general, 4=character, 9=rating
3. char_ip_map.json - OPTIONAL - character names as keys
4. booru_characters.csv - OPTIONAL - full character data with aliases and skins

All character tags are normalized to lowercase with underscores.
"""

import ast
import csv
import json
import re
from pathlib import Path
from typing import Any

from caption_pipeline.utils.logging_utils import log

_TAG_CACHE: dict[str, Any] = {}
_CHARACTER_DATA: dict[str, dict[str, Any]] = {}


def load_tag_databases() -> tuple[list[str], list[str]]:
    """
    Load general and character tag databases from all sources.

    Returns:
        Tuple of (general_tags, character_tags) - both sorted lists
    """
    cache_key = "tag_databases"
    if cache_key in _TAG_CACHE:
        return _TAG_CACHE[cache_key]

    general_tags: set[str] = set()
    character_tags: set[str] = set()

    # ============================================================
    # Source 1: tags_v0.9_13k.json (REQUIRED)
    # ============================================================
    # Contains ALL danbooru tags in tag_map with tag_split indicating
    # where general tags end and character tags begin
    pixai_path = Path("./tags_v0.9_13k.json")
    if pixai_path.exists():
        try:
            with pixai_path.open("r") as f:
                data = json.load(f)
                if "tag_map" in data and "tag_split" in data:
                    tag_split = data["tag_split"].get("gen_tag_count", 0)
                    tag_list = list(data["tag_map"].keys())

                    if tag_split > 0 and tag_split <= len(tag_list):
                        general_tags.update(tag_list[:tag_split])
                        character_tags.update(tag_list[tag_split:])
                        log.debug(
                            f"Split {len(tag_list)} tags from tags_v0.9_13k.json: "
                            f"{tag_split} general, {len(tag_list) - tag_split} character"
                        )
                    else:
                        raise ValueError(
                            f"Invalid tag_split value: {tag_split}. "
                            f"Expected between 1 and {len(tag_list)}"
                        )
                else:
                    raise KeyError("Missing 'tag_map' or 'tag_split' in tags_v0.9_13k.json")
        except Exception as e:
            log.error(f"Failed to load tags_v0.9_13k.json: {e}")
            raise
    else:
        raise FileNotFoundError(f"tags_v0.9_13k.json not found at {pixai_path}")

    # ============================================================
    # Source 2: selected_tags.csv (OPTIONAL)
    # ============================================================
    # Headers: tag_id, name, category, count
    # Categories: 0=general, 4=character, 9=rating
    wd_path = Path("./selected_tags.csv")
    if wd_path.exists():
        try:
            with wd_path.open("r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    name = row.get("name", "").strip()
                    category = row.get("category", "").strip()
                    if not name:
                        continue
                    if category == "0":  # general
                        general_tags.add(name)
                    elif category == "4":  # character
                        character_tags.add(name)
                    # category 9 = rating (ignored for tags)
        except Exception as e:
            log.warning(f"Failed to load selected_tags.csv: {e}")
    else:
        log.warning(f"selected_tags.csv not found at {wd_path}")

    # ============================================================
    # Source 3: char_ip_map.json (OPTIONAL)
    # ============================================================
    # Keys are character names
    char_ip_path = Path("./char_ip_map.json")
    if char_ip_path.exists():
        try:
            with char_ip_path.open("r") as f:
                data = json.load(f)
                for key in data.keys():
                    if key:
                        character_tags.add(key)
        except Exception as e:
            log.warning(f"Failed to load char_ip_map.json: {e}")
    else:
        log.warning(f"char_ip_map.json not found at {char_ip_path}")

    # ============================================================
    # Source 4: booru_characters.csv (OPTIONAL)
    # ============================================================
    # Contains character tags + aliases + skins with full data
    booru_path = Path("./booru_characters.csv")
    if booru_path.exists():
        try:
            with booru_path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Main tag
                    tag = row.get("tag", "").strip()
                    if tag:
                        character_tags.add(tag)

                    # Aliases
                    aliases = row.get("aliases", "").strip()
                    if aliases:
                        for alias in aliases.split(","):
                            alias = alias.strip()
                            if alias:
                                character_tags.add(alias)

                    # Skins
                    skins = row.get("skins", "").strip()
                    if skins:
                        for skin in skins.split(","):
                            skin = skin.strip()
                            if skin:
                                character_tags.add(skin)
        except Exception as e:
            log.warning(f"Failed to load booru_characters.csv: {e}")
    else:
        log.warning(f"booru_characters.csv not found at {booru_path}")

    # Convert to lists and normalize (lowercase with underscores)
    general_list = sorted([tag.lower().replace(" ", "_") for tag in general_tags if tag])
    character_list = sorted([tag.lower().replace(" ", "_") for tag in character_tags if tag])

    log.info(f"Loaded {len(general_list)} general tags and {len(character_list)} character tags")

    _TAG_CACHE[cache_key] = (general_list, character_list)
    return general_list, character_list


def load_character_tags_only() -> set[str]:
    """
    Convenience function to load ONLY character tags from all sources.

    Returns:
        Set of character tags
    """
    _, character_tags = load_tag_databases()
    return set(character_tags)


def load_general_tags_only() -> set[str]:
    """
    Convenience function to load ONLY general tags from all sources.

    Returns:
        Set of general tags
    """
    general_tags, _ = load_tag_databases()
    return set(general_tags)


def get_cached_tags() -> tuple[list[str], list[str]] | None:
    """
    Get cached tags without reloading.

    Returns:
        Tuple of (general_tags, character_tags) if cached, else None
    """
    cache_key = "tag_databases"
    return _TAG_CACHE.get(cache_key)


def clear_tag_cache() -> None:
    """Clear the tag cache."""
    cache_key = "tag_databases"
    if cache_key in _TAG_CACHE:
        del _TAG_CACHE[cache_key]
        log.debug("Tag cache cleared")


def load_character_data() -> dict[str, dict[str, Any]]:
    """
    Load character data from booru_characters.csv.

    This loads the full character database including:
    - Popular tags
    - Descriptions
    - Skins
    - Aliases
    - Parent relationships

    The data is cached after the first load.

    Returns:
        Dict mapping character tag -> row data
    """
    global _CHARACTER_DATA

    if _CHARACTER_DATA:
        return _CHARACTER_DATA

    csv_path = Path("./booru_characters.csv")

    if not csv_path.exists():
        log.warning(f"Character data file not found: {csv_path}")
        return {}

    def parse_list(value: str) -> list[str]:
        """Parse a string into a list, handling both Python literals and comma-separated."""
        if not value or value.strip() == "":
            return []
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            # Fallback: split by comma
            return [v.strip() for v in value.split(",") if v.strip()]

    try:
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                tag = row.get("tag", "").strip()
                if not tag:
                    continue

                _CHARACTER_DATA[tag] = {
                    "tag": tag,
                    "type": row.get("type", ""),
                    "parent_tag": row.get("parent_tag", ""),
                    "aliases": parse_list(row.get("aliases", "[]")),
                    "skins": parse_list(row.get("skins", "[]")),
                    "companions": parse_list(row.get("companions", "[]")),
                    "popular_tags": parse_list(row.get("popular_tags", "[]")),
                    "description": row.get("description", ""),
                }

                # Also index aliases
                for alias in _CHARACTER_DATA[tag]["aliases"]:
                    if alias:
                        _CHARACTER_DATA[alias] = _CHARACTER_DATA[tag]

                # Also index skins
                for skin in _CHARACTER_DATA[tag]["skins"]:
                    if skin:
                        _CHARACTER_DATA[skin] = _CHARACTER_DATA[tag]

        log.debug(f"Loaded {len(_CHARACTER_DATA)} character entries from {csv_path}")

    except Exception as e:
        log.error(f"Failed to load character data: {e}")
        return {}

    return _CHARACTER_DATA


def query_character(tag: str) -> dict[str, Any] | None:
    """
    Query character data by tag.

    Args:
        tag: The character tag to look up (normalized, lowercase with underscores)

    Returns:
        Dictionary with character data, or None if not found.

        Example:
        {
            "tag": "akekuri_(arknights)",
            "type": "character",
            "parent_tag": "",
            "aliases": ["akekuri", "ak_ke_ku_ri"],
            "skins": ["summer_akekuri_(arknights)"],
            "companions": [],
            "popular_tags": ["(5)", "arknights", "ak_ke_ku_ri"],
            "description": "A mascot character from Arknights..."
        }
    """
    data = load_character_data()
    return data.get(tag)


def query_character_field(tag: str, field: str) -> Any:
    """
    Query a specific field from character data.

    Args:
        tag: The character tag to look up
        field: The field name (e.g., "popular_tags", "description")

    Returns:
        The field value, or None if not found
    """
    data = query_character(tag)
    if data:
        return data.get(field)
    return None


def get_character_popular_tags(char_name: str) -> list[str]:
    """
    Get popular tags for a character.

    Args:
        char_name: The character tag to look up

    Returns:
        List of popular tags, or empty list if not found
    """
    data = query_character(char_name)
    if data:
        return data.get("popular_tags", [])
    return []


def get_character_description(char_name: str) -> str:
    """
    Get description for a character.

    Args:
        char_name: The character tag to look up

    Returns:
        Description string, or empty string if not found
    """
    data = query_character(char_name)
    if data:
        return data.get("description", "")
    return ""


def get_display_name(tag: str) -> str:
    """
    Get the display name for a character tag, resolving skins to their parent.

    For regular characters: returns the tag itself
    For skins: returns the parent character name
    For aliases: returns the canonical/parent name

    Args:
        tag: The character tag to resolve (normalized, lowercase with underscores)

    Returns:
        The resolved display name
    """
    data = query_character(tag)

    if not data:
        return tag

    tag_type = data.get("type", "").lower()
    parent_tag = data.get("parent_tag", "")

    # For skins and aliases, resolve to parent
    if tag_type in ("skin", "alias") and parent_tag:
        return parent_tag

    # Regular character or unknown type
    return tag


def get_parent_tag(tag: str) -> str | None:
    """
    Get the parent tag for a character, if it exists.

    Args:
        tag: The character tag to check

    Returns:
        The parent tag, or None if no parent exists
    """
    data = query_character(tag)

    if not data:
        return None

    return data.get("parent_tag") or None


def is_skin(tag: str) -> bool:
    """
    Check if a tag represents a skin.

    Args:
        tag: The character tag to check

    Returns:
        True if the tag is a skin, False otherwise
    """
    data = query_character(tag)

    if not data:
        return False

    return data.get("type", "").lower() == "skin"


def is_alias(tag: str) -> bool:
    """
    Check if a tag represents an alias.

    Args:
        tag: The character tag to check

    Returns:
        True if the tag is an alias, False otherwise
    """
    data = query_character(tag)

    if not data:
        return False

    return data.get("type", "").lower() == "alias"


def get_character_info(tag: str) -> dict[str, Any]:
    """
    Get formatted character info for display/logging.

    Args:
        tag: The character tag to look up

    Returns:
        Dictionary with formatted info:
        {
            "name": "akekuri_(arknights)",
            "display_name": "akekuri_(arknights)",
            "type": "character",
            "parent": None,
            "aliases": ["akekuri", "ak_ke_ku_ri"],
            "skins": ["summer_akekuri_(arknights)"],
            "popular_tags_count": 3,
            "has_description": True,
        }
    """
    data = query_character(tag)

    if not data:
        return {
            "name": tag,
            "display_name": tag,
            "type": "unknown",
            "parent": None,
            "aliases": [],
            "skins": [],
            "popular_tags_count": 0,
            "has_description": False,
        }

    return {
        "name": tag,
        "display_name": get_display_name(tag),
        "type": data.get("type", "unknown"),
        "parent": data.get("parent_tag") or None,
        "aliases": data.get("aliases", []),
        "skins": data.get("skins", []),
        "popular_tags_count": len(data.get("popular_tags", [])),
        "has_description": bool(data.get("description", "")),
    }


def search_characters(query: str, limit: int = 10) -> list[dict[str, Any]]:
    """
    Search for characters by tag name (partial match).

    Args:
        query: Search string (case-insensitive)
        limit: Maximum number of results to return

    Returns:
        List of character data dictionaries matching the query
    """
    data = load_character_data()
    results: list[dict[str, Any]] = []

    query_lower = query.lower().replace(" ", "_")

    for tag, info in data.items():
        # Skip aliases and skins (only show main entries)
        if info.get("type", "").lower() in ("alias", "skin"):
            continue

        # Check if query matches tag
        if query_lower in tag.lower():
            results.append(info)
            if len(results) >= limit:
                break

    return results


def get_all_characters() -> list[dict[str, Any]]:
    """
    Get all main character entries (excludes aliases and skins).

    Returns:
        List of all character data dictionaries
    """
    data = load_character_data()
    results: list[dict[str, Any]] = []

    for tag, info in data.items():
        # Skip aliases and skins
        if info.get("type", "").lower() in ("alias", "skin"):
            continue
        results.append(info)

    return sorted(results, key=lambda x: x.get("tag", ""))


def get_character_names() -> list[str]:
    """
    Get all character tag names (main entries only).

    Returns:
        List of character tag names
    """
    data = load_character_data()
    results: list[str] = []

    for tag, info in data.items():
        if info.get("type", "").lower() not in ("alias", "skin"):
            results.append(tag)

    return sorted(results)


def get_character_count_from_tag_confidences(tags: dict[str, float]) -> int:
    """
    Extract the total character count from scored count tags (1boy, 2girls, etc.).

    Count tags are normalized (lowercase, no spaces).
    Examples: 1boy, 2girls, 3boys, multiple girls, etc.

    Args:
        tags: Dictionary of tag -> confidence score

    Returns:
        Total number of characters indicated by count tags

    Examples:
        {"1boy": 0.95, "1girl": 0.90} → 2
        {"2boys": 0.85} → 2
        {"multiple girls": 0.70} → 0 (ambiguous, can't determine exact count)
        {"1boy": 0.95, "2girls": 0.88} → 3
    """
    count_patterns = [
        (r"^(\d+)boys?$", 1),  # 1boy, 2boys, 3boys
        (r"^(\d+)girls?$", 1),  # 1girl, 2girls, 3girls
        (r"^(\d+)others?$", 1),  # 1other, 2others
    ]

    total = 0

    for tag in tags.keys():
        tag_lower = tag.lower().strip()

        # Check for solo (indicates exactly 1 character)
        if tag_lower == "solo":
            return 1

        # Check for multiple (ambiguous count, we'll return 0 and let other tags decide)
        if tag_lower.startswith("multiple "):
            # "multiple boys", "multiple girls" - count is ambiguous
            continue

        # Try to extract count from tags like 1boy, 2girls
        for pattern, _ in count_patterns:
            match = re.match(pattern, tag_lower)
            if match:
                count = int(match.group(1))
                total += count
                break

    return total


def resolve_character_tags(
    user_character_tags: list[str],
    ai_character_tags: list[str],
    count: int,
    allow_ai: bool = True,
    threshold: float | None = None,
    context_name: str | None = None,
    all_tags: list[str] | None = None,
) -> list[str]:
    """
    Resolve character tags based on the count from count tags.
    
    Priority:
    1. User-provided character tags (from hints)
    2. AI-inferenced character tags (if allow_ai is True and more characters are needed)
    
    Args:
        user_character_tags: Character tags from user hints
        ai_character_tags: Character tags from AI inference (already filtered by threshold)
        count: Number of characters expected (from count tags)
        allow_ai: Whether to use AI-inferred characters
        threshold: The threshold used for AI character filtering (for logging)
        context_name: Optional context name for logging
        all_tags: Optional list of all tags to check for special tags
    
    Returns:
        List of resolved character tags (limited to 'count' items)
    """
    SPECIAL_TAGS = {"original", "borrowed_character"}
    
    if count < 0:
        raise ValueError(f"Character count cannot be negative: {count}")
    
    if count == 0:
        return []
    
    resolved = []
    
    # User characters first
    for tag in user_character_tags:
        if tag not in resolved:
            resolved.append(tag)
            if len(resolved) >= count:
                return resolved
    
    # AI characters if needed
    if allow_ai and len(resolved) < count:
        needed = count - len(resolved)
        if len(ai_character_tags) < needed:
            name_str = f" for {context_name}" if context_name else ""
            threshold_str = f" (threshold: {threshold})" if threshold is not None else ""
            
            # Check if special tags are present
            has_special_tag = any(tag in SPECIAL_TAGS for tag in (all_tags or []))
            note_str = " (note: 'original' or 'borrowed_character' present - may be an unnamed/original character)" if has_special_tag else ""
            
            log.warning(
                f"Insufficient AI character tags{name_str}: "
                f"need {needed} more characters, "
                f"but only {len(ai_character_tags)} AI tags available{threshold_str}{note_str}"
            )
        
        for tag in ai_character_tags:
            if tag not in resolved:
                resolved.append(tag)
                if len(resolved) >= count:
                    return resolved
    
    # Mismatch warning
    if len(resolved) < count:
        name_str = f" for {context_name}" if context_name else ""
        
        # Check if special tags are present
        has_special_tag = any(tag in SPECIAL_TAGS for tag in (all_tags or []))
        note_str = " (note: 'original' or 'borrowed_character' present - this may be an unnamed/original character)" if has_special_tag else ""
        
        log.warning(
            f"Character count mismatch{name_str}: {count} characters expected from count tags, "
            f"but only {len(resolved)} character tags available "
            f"(user: {len(user_character_tags)}, AI: {len(ai_character_tags)}){note_str}"
        )
    
    return resolved


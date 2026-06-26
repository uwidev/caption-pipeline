"""
Tag Database Utilities - Centralized loading of all danbooru tag sources.

This module loads and caches tags from all sources:
1. tags_v0.9_13k.json (PixAI) - REQUIRED - contains both general and character tags
2. selected_tags.csv (WD14) - OPTIONAL - categories: 0=general, 4=character, 9=rating
3. char_ip_map.json - OPTIONAL - character names as keys
4. booru_characters.csv - OPTIONAL - full character data with aliases and skins

All character tags are normalized to lowercase with underscores.
"""

import csv
import json
from pathlib import Path

from loguru import logger

_TAG_CACHE: dict[str, any] = {}


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
                        logger.debug(
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
            logger.error(f"Failed to load tags_v0.9_13k.json: {e}")
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
            logger.warning(f"Failed to load selected_tags.csv: {e}")
    else:
        logger.warning(f"selected_tags.csv not found at {wd_path}")

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
            logger.warning(f"Failed to load char_ip_map.json: {e}")
    else:
        logger.warning(f"char_ip_map.json not found at {char_ip_path}")

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
            logger.warning(f"Failed to load booru_characters.csv: {e}")
    else:
        logger.warning(f"booru_characters.csv not found at {booru_path}")

    # Convert to lists and normalize (lowercase with underscores)
    general_list = sorted([tag.lower().replace(" ", "_") for tag in general_tags if tag])
    character_list = sorted([tag.lower().replace(" ", "_") for tag in character_tags if tag])

    logger.info(f"Loaded {len(general_list)} general tags and {len(character_list)} character tags")

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
        logger.debug("Tag cache cleared")

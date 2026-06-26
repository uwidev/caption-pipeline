"""
Character extraction with type-safe source tracking.

This module provides precise character identification with source tracking:
- USER_HINTED: User explicitly used "character:" prefix
- DATABASE_LOOKUP: Found in booru_characters.csv
- ALIAS_RESOLVED: Resolved from an alias to canonical name
- SKIN_AS_CHARACTER: Skin entry treated as character itself
- EXTRACTED: Generic extraction (fallback)

Key features:
- Preserves skin tags (doesn't resolve to parent)
- Tracks source for debugging and prioritization
- Handles "character:" prefix authoritatively
- Type-safe with Enums and dataclasses
- Uses centralized tag database from tag_db.py
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Self

from loguru import logger

from caption_pipeline.utils.tag_db import load_tag_databases, load_character_tags_only


class CharacterSource(Enum):
    """
    Tracks where a character tag originated.
    
    This is critical for understanding how to handle the character:
    - USER_HINTED: User explicitly tagged this as a character (with "character:" prefix)
    - DATABASE_LOOKUP: Tag was found in booru_characters.csv via AI inference
    - ALIAS_RESOLVED: Tag resolved from an alias to a canonical name
    - SKIN_AS_CHARACTER: Tag is a skin entry, treated as the character itself
    - EXTRACTED: Generic extraction (fallback)
    """
    USER_HINTED = auto()
    DATABASE_LOOKUP = auto()
    ALIAS_RESOLVED = auto()
    SKIN_AS_CHARACTER = auto()
    EXTRACTED = auto()
    
    def is_hinted(self) -> bool:
        """Check if this was user-hinted."""
        return self == CharacterSource.USER_HINTED
    
    def is_authoritative(self) -> bool:
        """Check if this should override other sources."""
        return self in {CharacterSource.USER_HINTED, CharacterSource.SKIN_AS_CHARACTER}


@dataclass(slots=True)
class CharacterData:
    """Database row data for a character."""
    tag: str
    type: str
    parent_tag: str | None
    aliases: list[str]
    skins: list[str]
    companions: list[str]
    popular_tags: list[str]
    description: str
    
    @classmethod
    def from_row(cls, row: dict[str, str]) -> Self:
        """Create CharacterData from CSV row."""
        import ast
        
        def parse_list(value: str) -> list[str]:
            if not value or value.strip() == "":
                return []
            try:
                return ast.literal_eval(value)
            except (ValueError, SyntaxError):
                # Fallback: split by comma
                return [v.strip() for v in value.split(",") if v.strip()]
        
        return cls(
            tag=row.get("tag", ""),
            type=row.get("type", ""),
            parent_tag=row.get("parent_tag") or None,
            aliases=parse_list(row.get("aliases", "[]")),
            skins=parse_list(row.get("skins", "[]")),
            companions=parse_list(row.get("companions", "[]")),
            popular_tags=parse_list(row.get("popular_tags", "[]")),
            description=row.get("description", ""),
        )
    
    def is_alias(self) -> bool:
        """Check if this entry is an alias."""
        return self.type.lower() == "alias"
    
    def is_skin(self) -> bool:
        """Check if this entry is a skin."""
        return self.type.lower() == "skin"
    
    def get_canonical(self) -> str:
        """Get the canonical tag (for aliases, this is the parent)."""
        if self.is_alias() and self.parent_tag:
            return self.parent_tag
        return self.tag


@dataclass(slots=True)
class CharacterEntry:
    """
    A character entry with source tracking.
    
    This preserves the exact tag the user provided while also tracking
    how we identified it as a character.
    """
    tag: str  # Canonical tag (e.g., "summer_akekuri_(arknights)")
    source: CharacterSource
    original_tag: str | None = None  # The tag before normalization
    parent_tag: str | None = None  # Parent tag if this is an alias/skin
    is_skin: bool = False  # Whether this represents a skin entry
    data: CharacterData | None = None  # Full database row if available
    
    @classmethod
    def from_hint(cls, tag: str, normalized: str | None = None) -> Self:
        """
        Create a character entry from a user hint.
        
        This is the most authoritative source.
        """
        return cls(
            tag=normalized or tag,
            source=CharacterSource.USER_HINTED,
            original_tag=tag,
        )
    
    @classmethod
    def from_database(cls, tag: str, data: CharacterData) -> Self:
        """
        Create a character entry from database lookup.
        
        Args:
            tag: The matched tag (could be alias, skin, or canonical)
            data: The database row for this tag
        """
        # Determine if this is a skin
        is_skin = data.is_skin()
        
        # For skins, the canonical tag is the skin itself, not the parent
        if is_skin:
            canonical = tag  # Keep the skin tag
            source = CharacterSource.SKIN_AS_CHARACTER
        elif data.is_alias():
            # Aliases resolve to parent
            canonical = data.get_canonical()
            source = CharacterSource.ALIAS_RESOLVED
        else:
            # Regular character
            canonical = data.tag
            source = CharacterSource.DATABASE_LOOKUP
        
        return cls(
            tag=canonical,
            source=source,
            original_tag=tag,
            parent_tag=data.parent_tag,
            is_skin=is_skin,
            data=data,
        )
    
    def get_display_name(self) -> str:
        """
        Get the display name for captioning.
        
        For skins, this is the skin tag itself.
        For resolved aliases, this is the parent tag.
        For others, this is the canonical tag.
        """
        return self.tag
    
    def get_popular_tags(self) -> list[str]:
        """Get popular tags from the database entry."""
        if self.data:
            return self.data.popular_tags
        return []
    
    def get_description(self) -> str:
        """Get description from the database entry."""
        if self.data:
            return self.data.description
        return ""
    
    def get_parent_tag(self) -> str | None:
        """Get the parent tag if this is a skin or alias."""
        return self.parent_tag


class CharacterDatabase:
    """
    Character database with query capabilities.
    
    Loads from booru_characters.csv and provides:
    - Exact match queries
    - Alias resolution
    - Skin detection
    - Tag existence checks
    """
    
    def __init__(self) -> None:
        """Initialize the character database by loading from booru_characters.csv."""
        self._db: dict[str, CharacterData] = {}
        self._load()
    
    def _load(self) -> None:
        """Load the CSV database."""
        booru_path = Path("./booru_characters.csv")
        
        if not booru_path.exists():
            logger.warning(f"Character database not found: {booru_path}")
            return
        
        try:
            with booru_path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    tag = row.get("tag", "").strip()
                    if not tag:
                        continue
                    
                    data = CharacterData.from_row(row)
                    self._db[tag] = data
                    
                    # Also index aliases
                    for alias in data.aliases:
                        alias = alias.strip()
                        if alias:
                            self._db[alias] = data
                    
                    # Also index skins
                    for skin in data.skins:
                        skin = skin.strip()
                        if skin:
                            self._db[skin] = data
            
            logger.debug(f"Loaded {len(self._db)} character entries from {booru_path}")
        
        except Exception as e:
            logger.error(f"Failed to load character database: {e}")
            raise
    
    def query(self, tag: str) -> CharacterData | None:
        """
        Query the database for a tag.
        
        Returns:
            CharacterData if found, None otherwise.
        """
        return self._db.get(tag)
    
    def get_canonical(self, tag: str) -> str | None:
        """
        Get the canonical tag for a tag (resolves aliases).
        
        Returns:
            Canonical tag, or None if not found.
        """
        data = self.query(tag)
        if data:
            return data.get_canonical()
        return None
    
    def is_character(self, tag: str) -> bool:
        """Check if a tag exists in the database."""
        return tag in self._db
    
    def get_all_tags(self) -> set[str]:
        """Get all tags in the database."""
        return set(self._db.keys())
    
    def __contains__(self, tag: str) -> bool:
        return self.is_character(tag)
    
    def __len__(self) -> int:
        return len(self._db)


# Global instances (lazy-loaded)
_DATABASE: CharacterDatabase | None = None
_CHARACTER_TAGS_CACHE: set[str] | None = None


def get_character_database() -> CharacterDatabase:
    """Get the global character database instance (lazy-loaded)."""
    global _DATABASE
    if _DATABASE is None:
        _DATABASE = CharacterDatabase()
    return _DATABASE


def get_character_tags_set() -> set[str]:
    """
    Get the set of all character tags from the centralized tag database.
    
    This uses the tag_db.py cache, avoiding duplicate loading.
    
    Returns:
        Set of character tags
    """
    global _CHARACTER_TAGS_CACHE
    if _CHARACTER_TAGS_CACHE is None:
        _CHARACTER_TAGS_CACHE = load_character_tags_only()
    return _CHARACTER_TAGS_CACHE


def normalize_character_tag(tag: str) -> str:
    """
    Normalize a character tag to database format.
    
    Database format: lowercase_with_underscores (e.g., "akekuri_(arknights)")
    User format: may have spaces (e.g., "akekuri (arknights)")
    
    Args:
        tag: The tag to normalize
        
    Returns:
        Normalized tag, or empty string if invalid.
    """
    if not tag:
        return ""

    # Remove "character:" prefix if present
    if tag.startswith("character:"):
        tag = tag[10:]

    # Convert to lowercase
    tag = tag.lower()

    # Convert spaces to underscores
    tag = tag.replace(" ", "_")

    # Remove any leading/trailing underscores
    return tag.strip("_ ")


class CharacterExtractor:
    """
    Character extraction with precise source tracking.
    
    Rules:
    1. Tags with "character:" prefix → ALWAYS treated as character (USER_HINTED)
    2. Tags found in database → treated as character (DATABASE_LOOKUP)
    3. All other tags → treated as general tags
    4. Skins → treated as the character itself, NOT resolved to parent
    5. Aliases → resolved to their canonical name (ALIAS_RESOLVED)
    
    Source priority:
    1. USER_HINTED (highest)
    2. SKIN_AS_CHARACTER
    3. ALIAS_RESOLVED / DATABASE_LOOKUP
    4. EXTRACTED (lowest)
    """
    
    def __init__(self) -> None:
        """Initialize the extractor with the character database and tag set."""
        self._db = get_character_database()
        self._character_tags = get_character_tags_set()
    
    def extract(
        self,
        tags: list[list[str]],
        *,
        remove_from_sections: bool = True,
    ) -> tuple[list[list[str]], list[CharacterEntry]]:
        """
        Extract character tags with source tracking.
        
        Args:
            tags: List of tag sections (0=prepended, 1=main, 2=nl)
            remove_from_sections: Remove character tags from sections
            
        Returns:
            Tuple of (modified_tags, character_entries)
            - modified_tags: Tags with character entries removed
            - character_entries: List of CharacterEntry objects
        """
        modified = [section[:] for section in tags]
        found_entries: dict[str, CharacterEntry] = {}
        
        for section_idx, section in enumerate(modified):
            if not section:
                continue
            
            tags_to_remove: list[str] = []
            
            for tag in section:
                # Try to extract character using precise rules
                entry = self._extract_single(tag)
                
                if entry:
                    # Only add if not already found (prefer existing)
                    if entry.tag not in found_entries:
                        found_entries[entry.tag] = entry
                        logger.debug(
                            f"Extracted character: '{tag}' -> '{entry.tag}' "
                            f"(source: {entry.source.name})"
                        )
                    else:
                        # Check if this entry is more authoritative
                        existing = found_entries[entry.tag]
                        if entry.source.is_authoritative() and not existing.source.is_authoritative():
                            found_entries[entry.tag] = entry
                            logger.debug(
                                f"Replaced existing character '{entry.tag}' with "
                                f"more authoritative source ({entry.source.name})"
                            )
                    
                    # Mark for removal from general tags
                    tags_to_remove.append(tag)
            
            # Remove marked tags
            if remove_from_sections and tags_to_remove:
                modified[section_idx] = [t for t in section if t not in tags_to_remove]
        
        # Convert to list with deterministic order
        def sort_key(entry: CharacterEntry) -> int:
            # User-hinted entries get highest priority
            if entry.source == CharacterSource.USER_HINTED:
                return 0
            elif entry.source == CharacterSource.SKIN_AS_CHARACTER:
                return 1
            elif entry.source == CharacterSource.ALIAS_RESOLVED:
                return 2
            else:
                return 3
        
        entries = sorted(found_entries.values(), key=sort_key)
        
        if entries:
            display_tags = [e.get_display_name() for e in entries]
            sources = [e.source.name for e in entries]
            logger.info(
                f"Extracted {len(entries)} characters: {', '.join(display_tags)} "
                f"(sources: {', '.join(sources)})"
            )
        
        return modified, entries
    
    def _extract_single(self, tag: str) -> CharacterEntry | None:
        """
        Extract a single character tag with precise rules.
        
        Rule 1: "character:" prefix → ALWAYS character (USER_HINTED)
        Rule 2: Database match → character entry (DATABASE_LOOKUP)
        Rule 3: Everything else → not a character
        
        For skins: Keep the skin tag, do NOT resolve to parent.
        For aliases: Resolve to parent, track as ALIAS_RESOLVED.
        """
        # Rule 1: User explicitly hinted this is a character
        if tag.startswith("character:"):
            # Extract the actual tag name
            actual_tag = tag[10:].strip()
            if actual_tag:
                # Normalize for consistency
                normalized = normalize_character_tag(actual_tag)
                if normalized:
                    return CharacterEntry.from_hint(normalized, normalized)
            return None
        
        # Rule 2: Check database and character tag set
        normalized = normalize_character_tag(tag)
        if not normalized:
            return None
        
        # First check if it's in the character tag set (from centralized DB)
        if normalized in self._character_tags:
            # Query the database for full data
            data = self._db.query(normalized)
            if data:
                # Create entry from database row
                # For skins, the tag stays as the skin tag
                # For aliases, we resolve to parent
                return CharacterEntry.from_database(normalized, data)
            else:
                # Found in character tag set but no data - create basic entry
                return CharacterEntry(
                    tag=normalized,
                    source=CharacterSource.DATABASE_LOOKUP,
                    original_tag=tag,
                )
        
        # Rule 3: Not a character
        return None
    
    def is_character_tag(self, tag: str) -> bool:
        """
        Check if a tag is a character tag.
        
        This follows the same rules as _extract_single:
        - "character:" prefix → True
        - In character tag set → True
        - Otherwise → False
        """
        # Rule 1: User explicitly hinted this is a character
        if tag.startswith("character:"):
            return True
        
        # Rule 2: Check character tag set
        normalized = normalize_character_tag(tag)
        if not normalized:
            return False
        
        return normalized in self._character_tags


# Convenience function for backward compatibility
def extract_characters_from_tags(
    tags: list[list[str]],
    *,
    remove_from_sections: bool = True,
) -> tuple[list[list[str]], list[str]]:
    """
    Extract character tags from a list of tag sections (backward compatibility).
    
    This wraps CharacterExtractor and returns only the character names as strings.
    
    Args:
        tags: List of tag sections
        remove_from_sections: Remove character tags from sections
        
    Returns:
        Tuple of (modified_tags, character_names)
    """
    extractor = CharacterExtractor()
    modified, entries = extractor.extract(tags, remove_from_sections=remove_from_sections)
    character_names = [e.tag for e in entries]
    return modified, character_names


# Convenience function for getting character data
def load_character_database() -> dict[str, dict[str, str]]:
    """
    Load character database as dict for backward compatibility.
    
    Returns:
        Dict mapping tag -> row data (as dict of strings)
    """
    db = get_character_database()
    
    # Convert CharacterData objects to dicts for compatibility
    result: dict[str, dict[str, str]] = {}
    for tag, data in db._db.items():
        result[tag] = {
            "tag": data.tag,
            "type": data.type,
            "parent_tag": data.parent_tag or "",
            "aliases": ", ".join(data.aliases),
            "skins": ", ".join(data.skins),
            "companions": ", ".join(data.companions),
            "popular_tags": str(data.popular_tags),
            "description": data.description,
        }
    
    return result

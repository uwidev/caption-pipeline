"""
ImageContext: The data container that flows through the pipeline.

Tags for an image uses underscore instead of spaces.

The only times it should be space is when reading from grounding hints,
in which they are immediately converted to underscores for internal use,
and when doing the final write.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

from PIL import Image

ORIGINAL_CHARACTER = ("original", "borrowed_character")


@dataclass(slots=True)
class ImageContext:
    """
    Container for image data and metadata flowing through the pipeline.

    Attributes:
        image_path: Path to the image file
        source_path: Original source path (for tracking)
        image_data: Loaded PIL Image (lazy-loaded)
        tags: Three sections of tags: [prepended, main, nl]
        original_tags: Complete original tags from all sections (preserved for reference)
        rating: Optional content rating (safe, questionable, explicit)
        character_tags: List of character tag names (normalized, lowercase with underscores)
        metadata: Additional metadata storage
        history: Processing history
        inferenced_tags: All tags from AI inference (tag -> confidence)
    """

    image_path: Path
    source_path: Path
    image_data: Image.Image | None = None
    tags: list[list[str]] = field(default_factory=lambda: [[], [], []])
    original_tags: list[list[str]] = field(default_factory=lambda: [[], [], []])  # Preserved originals
    rating: str | None = None
    character_tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    history: list[str] = field(default_factory=list)
    inferenced_tags: dict[str, float] | None = None

    # Private state
    _is_modified: bool = field(default=False, init=False)

    def get_tags(self, section: int = 1) -> list[str]:
        """Get tags from a specific section (0=prepended, 1=main, 2=nl)."""
        if section < 0 or section >= len(self.tags):
            return []
        return self.tags[section]

    def set_tags(self, tags: list[str], section: int = 1) -> None:
        """Set tags for a specific section."""
        while len(self.tags) <= section:
            self.tags.append([])
        self.tags[section] = tags.copy()
        self._is_modified = True

    def add_tags(self, tags: str | list[str], section: int = 1, position: int = -1) -> None:
        """
        Add tags to a section.

        Args:
            tags: Tag or list of tags to add
            section: Section to add to (0=prepended, 1=main, 2=nl)
            position: -1=append, 0=prepend, >0=insert at index
        """
        if isinstance(tags, str):
            tag_list = [tags]
        else:
            tag_list = tags.copy()

        current = self.tags[section]

        match position:
            case -1:  # Append
                current.extend(tag_list)
            case 0:  # Prepend
                self.tags[section] = tag_list + current
            case pos:  # Insert at specific position
                self.tags[section] = current[:pos] + tag_list + current[pos:]

        self._is_modified = True

    def remove_tags(self, tags: list[str], section: int = 1) -> None:
        """Remove specific tags from a section."""
        tag_set = set(tags)
        self.tags[section] = [t for t in self.tags[section] if t not in tag_set]
        self._is_modified = True

    def get_full_caption(self, delimiter: str = " ||| ") -> str:
        """Get the full caption with sections joined by delimiter."""
        sections = [", ".join(section) for section in self.tags if section]
        return delimiter.join(sections)

    def load_image(self) -> Image.Image:
        """Lazy load the image data."""
        if self.image_data is None:
            self.image_data = Image.open(self.image_path)
        return self.image_data

    def save_image(self, output_path: Path | None = None) -> None:
        """Save the current image data if modified."""
        if self.image_data is not None and self._is_modified:
            path = output_path or self.image_path
            self.image_data.save(path)
            self.image_path = path
            self._is_modified = False

    def copy(self) -> Self:
        """Create a shallow copy of the context."""
        return ImageContext(
            image_path=self.image_path,
            source_path=self.source_path,
            image_data=self.image_data,
            tags=[section.copy() for section in self.tags],
            original_tags=[section.copy() for section in self.original_tags],
            rating=self.rating,
            character_tags=self.character_tags.copy(),
            metadata=self.metadata.copy(),
            history=self.history.copy(),
            inferenced_tags=(
                self.inferenced_tags.copy() if self.inferenced_tags is not None else None
            ),
        )

    def add_history(self, step_name: str) -> None:
        """Add a step to the processing history."""
        self.history.append(step_name)

    # ===== Character Helpers =====

    def has_characters(self) -> bool:
        """Check if any character tags exist."""
        return bool(self.character_tags)

    def has_unnamed_character(self) -> bool:
        """
        Check if there's an unnamed/original character.

        Unnamed characters are designated by having no valid character tags while having
        'original' or 'borrowed_character' in section 1 of tags.
        """
        return not self.character_tags and any(
            (tag in self.get_tags(1) for tag in ORIGINAL_CHARACTER)
        )

    def get_character_tags(self) -> list[str]:
        """Get the list of character tag names."""
        return self.character_tags.copy()

    def set_characters(self, tags: list[str]) -> None:
        """Set character tags and their source."""
        self.character_tags = tags.copy()

    def clear_characters(self) -> None:
        """Clear all character tags."""
        self.character_tags = []

    # ===== Original Tags Helpers =====

    def get_original_tags(self, section: int = -1) -> list[str] | list[list[str]]:
        """
        Get the original tags.

        Args:
            section: Section to get (-1 = all sections, 0-2 = specific section)

        Returns:
            List of tags for a specific section, or list of all sections
        """
        if section == -1:
            return [section.copy() for section in self.original_tags]
        if section < 0 or section >= len(self.original_tags):
            return []
        return self.original_tags[section].copy()

    def set_original_tags(self, tags: list[list[str]]) -> None:
        """Set the original tags."""
        self.original_tags = [section.copy() for section in tags]

    def get_original_flat(self) -> list[str]:
        """Get all original tags flattened (sections 0 and 1 only)."""
        result = []
        for section in [0, 1]:
            if section < len(self.original_tags):
                result.extend(self.original_tags[section])
        return result

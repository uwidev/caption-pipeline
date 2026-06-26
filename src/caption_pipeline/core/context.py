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


@dataclass(slots=True)
class ImageContext:
    """
    Container for image data and metadata flowing through the pipeline.
    
    Attributes:
        image_path: Path to the image file
        source_path: Original source path (for tracking)
        image_data: Loaded PIL Image (lazy-loaded)
        tags: Three sections of tags: [prepended, main, nl]
        rating: Optional content rating (safe, questionable, explicit)
        character_entries: List of CharacterEntry objects with source tracking
        metadata: Additional metadata storage
        history: Processing history
        inferenced_tags: All tags from AI inference (tag -> confidence)
    """
    
    image_path: Path
    source_path: Path
    image_data: Image.Image | None = None
    tags: list[list[str]] = field(
        default_factory=lambda: [[], [], []]
    )
    rating: str | None = None
    character_entries: list[Any] = field(default_factory=list)  # CharacterEntry objects
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
    
    def add_tags(
        self, 
        tags: str | list[str], 
        section: int = 1, 
        position: int = -1
    ) -> None:
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
            rating=self.rating,
            character_entries=self.character_entries.copy(),
            metadata=self.metadata.copy(),
            history=self.history.copy(),
            inferenced_tags=(
                self.inferenced_tags.copy() 
                if self.inferenced_tags is not None 
                else None
            ),
        )
    
    def add_history(self, step_name: str) -> None:
        """Add a step to the processing history."""
        self.history.append(step_name)
    
    # ===== Character Entry Helpers =====
    
    def get_character_tags(self) -> list[str]:
        """Get canonical character tags."""
        return [e.tag for e in self.character_entries]
    
    def get_character_data(self) -> dict[str, dict[str, str]]:
        """
        Get character data from entries.
        
        Returns:
            Dict mapping character tag -> data dict
        """
        result: dict[str, dict[str, str]] = {}
        for entry in self.character_entries:
            if entry.data:
                result[entry.tag] = {
                    "tag": entry.data.tag,
                    "type": entry.data.type,
                    "parent_tag": entry.data.parent_tag or "",
                    "aliases": ", ".join(entry.data.aliases),
                    "skins": ", ".join(entry.data.skins),
                    "companions": ", ".join(entry.data.companions),
                    "popular_tags": str(entry.data.popular_tags),
                    "description": entry.data.description,
                }
        return result
    
    def has_characters(self) -> bool:
        """Check if any character entries exist."""
        return bool(self.character_entries)

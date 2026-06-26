"""
Danbooru character database utilities.

This module provides a class for loading and querying character data
from a CSV database. Supports aliases and skin references.

Usage:
    from caption_pipeline.utils.booru_characters import DanbooruCharacters
    
    db = DanbooruCharacters("booru_characters.csv")
    character_data = db.query("Albedo")
"""

import csv
from pathlib import Path


class DanbooruCharacters:
    """Load and query Danbooru character database from CSV."""

    def __init__(self, filename: str | Path = "booru_characters.csv") -> None:
        """
        Initialize the DanbooruCharacters class by loading the CSV file.

        Args:
            filename: Path to the CSV file (default: "booru_characters.csv")
        """
        self.db: dict[str, dict[str, str]] = {}
        self.headers: list[str] = []
        self.load_csv(filename)

    def load_csv(self, filename: str | Path) -> None:
        """
        Load CSV file and store it as a dictionary.

        Args:
            filename: Path to the CSV file
        """
        try:
            with open(filename, "r", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                self.headers = reader.fieldnames or []

                for row in reader:
                    # First column is the key (assuming first header is 'tag')
                    tag_key = row[self.headers[0]]

                    # Create a dictionary mapping the rest of the columns
                    row_data: dict[str, str] = {}
                    for header in self.headers[1:]:
                        row_data[header] = row[header]

                    # Store in database
                    self.db[tag_key] = row_data

        except FileNotFoundError:
            print(f"Error: File '{filename}' not found.")
            raise
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            raise

    def query(
        self, 
        tag: str, 
        replace_spaces: bool = True, 
        follow_aliases: bool = True
    ) -> dict[str, str] | None:
        """
        Query the database for a specific tag.

        If the tag is of type 'alias' or 'skin', it will recursively query
        using the parent_tag field.

        Args:
            tag: The tag to search for
            replace_spaces: If True, replaces spaces with underscores in the query tag
            follow_aliases: If True, follows alias/skin references to parent tags

        Returns:
            Dictionary of the rest of the row's columns, or None if not found
        """
        if replace_spaces:
            tag = tag.replace(" ", "_")

        result = self.db.get(tag)

        # If we found a result and we should follow aliases/skins
        if result and follow_aliases:
            tag_type = result.get("type", "")
            parent_tag = result.get("parent_tag", "")

            # If type is alias or skin, and there's a parent_tag, query recursively
            if tag_type.lower() in ["alias", "skin"] and parent_tag:
                # Recursively query the parent tag
                return self.query(parent_tag, replace_spaces=False, follow_aliases=True)

        return result

    def get_all_tags(self) -> list[str]:
        """
        Get all available tags.

        Returns:
            List of all tag keys
        """
        return list(self.db.keys())

    def tag_exists(self, tag: str) -> bool:
        """
        Check if a tag exists in the database.

        Args:
            tag: The tag to check

        Returns:
            True if tag exists, False otherwise
        """
        return tag in self.db

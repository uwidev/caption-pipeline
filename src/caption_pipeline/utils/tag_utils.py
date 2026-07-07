"""
Tag sorting utilities for the tools package.
"""


def sort_key_by_object(tag: str) -> tuple[str, str]:
    """
    Generate a sort key that prioritizes the object (last word) first.

    Example:
        'blue_hair' -> ('hair', 'blue')
        'red_eyes' -> ('eyes', 'red')
        'standing' -> ('standing', '')

    This is used to sort tags by object within categories.
    """
    parts = tag.split()
    if len(parts) >= 2:
        # Convert underscores to spaces first for consistent sorting
        normalized_parts = [p.replace("_", " ") for p in parts]
        return (normalized_parts[-1], " ".join(normalized_parts[:-1]))
    else:
        return (tag.replace("_", " "), "")


def sort_tags_by_object(tags: set[str]) -> list[str]:
    """
    Sort tags by the object (last word) first, then modifier.

    Example:
        ['blue hair', 'brown hair', 'red eyes'] -> ['blue hair', 'brown hair', 'red eyes']
        (sorted by last word: hair, then eyes; within same last word, by modifier).

    If a tag has no space, it's sorted by the whole tag as a single key.

    Args:
        tags: Set of tags (normalized with spaces).

    Returns:
        Sorted list of tags.
    """
    return sorted(tags, key=sort_key_by_object)

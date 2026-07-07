"""
Tag sorting utilities for the tools package.
"""


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

    def key_func(tag: str) -> tuple[str, str]:
        parts = tag.split()
        if len(parts) >= 2:
            # Use last token as primary key, the rest as secondary
            return (parts[-1], " ".join(parts[:-1]))
        else:
            return (tag, "")

    return sorted(tags, key=key_func)

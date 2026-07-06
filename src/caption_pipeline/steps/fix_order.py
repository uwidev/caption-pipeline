"""
FixOrderStep: Reorder tags in a section according to a specified ordering mode.

Modes:
- category: Reorder by semantic category (using the tag cache). Tags are grouped by
  their semantic category (character count, rating, body parts, etc.) and sorted
  alphabetically within each group.
- rating_character: Reorder as: rating tag (if any), then character tags, then
  everything else (preserving original order for the rest).

This step updates the context with the reordered tags. It may trigger LLM calls
for tags not yet present in the tag cache.
"""

import re
from typing import Literal

from caption_pipeline.core.context import ImageContext
from caption_pipeline.core.help import step_help
from caption_pipeline.core.step import PipelineStep
from caption_pipeline.utils.logging_utils import log, section
from caption_pipeline.utils.tag_cache import TagCategoryCache

CATEGORY_ORDER = [
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
]

RATING_TAGS = {"general", "sensitive", "questionable", "explicit", "safe", "suggestive", "nsfw"}

# Patterns for count tags
COUNT_PATTERNS = [
    r"^(\d+)boys?$",      # 1boy, 2boys, 3boys
    r"^(\d+)girls?$",     # 1girl, 2girls, 3girls
    r"^(\d+)others?$",    # 1other, 2others
]


def is_count_tag(tag: str) -> bool:
    """Check if a tag is a count tag (e.g., 1girl, 2boys, solo)."""
    tag_lower = tag.lower().strip()
    if tag_lower == "solo":
        return True
    for pattern in COUNT_PATTERNS:
        if re.match(pattern, tag_lower):
            return True
    return False


@step_help(
    name="fix:order",
    description="Reorder tags in a specific section according to a mode.",
    long_description="""This step reorders tags in the given section.

Modes:
- category: Reorder by semantic category (using the tag cache). Tags are grouped by
  category and sorted alphabetically within each group. This is useful for consistent
  ordering across training examples. Unseen tags are classified via the LLM and
  added to the cache.

  Rating and character tags are automatically identified and pre‑categorized before
  the LLM call, so they are never sent to the LLM.

- rating_character: Reorder as: rating tag (if any), then count tags, then character tags,
  then everything else (preserving original order for the rest). This matches the
  ordering used by format:join.

The step uses the persistent tag cache. If a tag is not yet in the cache, it will
be classified on the fly via the LLM (Ollama).""",
    options=[
        {
            "flag": "--section INT",
            "help": "Section to reorder (0=prepended, 1=main)",
            "default": "1",
        },
        {
            "flag": "--mode {category,rating_character}",
            "help": "Ordering mode",
            "default": "category",
        },
        {
            "flag": "--model NAME",
            "help": "Ollama model for classification",
            "default": "huihui_ai/phi4-abliterated:14b",
        },
        {
            "flag": "--url URL",
            "help": "Ollama API URL",
            "default": "http://localhost:11434/api/chat",
        },
        {"flag": "--temperature FLOAT", "help": "Temperature for generation", "default": "0.6"},
        {"flag": "--timeout INT", "help": "Request timeout in seconds", "default": "120"},
        {
            "flag": "--batch-size INT",
            "help": "Batch size for LLM requests (-1 = all at once)",
            "default": "-1",
        },
    ],
    example="fix:order --section 1 --mode rating_character",
)
class FixOrderStep(PipelineStep):
    def __init__(
        self,
        section: int = 1,
        order_mode: Literal["category", "rating_character"] = "category",
        model: str = "huihui_ai/phi4-abliterated:14b",
        ollama_url: str = "http://localhost:11434/api/chat",
        temperature: float = 0.6,
        timeout: int = 120,
        batch_size: int = 20,
    ) -> None:
        self.section = section
        self.order_mode = order_mode
        self.model = model
        self.ollama_url = ollama_url
        self.temperature = temperature
        self.timeout = timeout
        self.batch_size = batch_size

    def name(self) -> str:
        return "fix:order"

    def validate(self, context: ImageContext) -> bool:
        return 0 <= self.section < len(context.tags) and bool(context.tags[self.section])

    def process(self, context: ImageContext) -> ImageContext | None:
        with section(f"Processing {context.image_path.name} - reordering section {self.section}"):
            result = context.copy()
            tags = context.get_tags(self.section)
            if not tags:
                log.debug("No tags to reorder")
                return result

            log.debug(f"[BEFORE] Original tags: {tags}")

            if self.order_mode == "category":
                ordered = self._order_by_category(tags, context)
            elif self.order_mode == "rating_character":
                ordered = self._order_rating_character(tags, context)
            else:
                raise ValueError(f"Unknown order_mode: {self.order_mode}")

            log.debug(f"[AFTER] Ordered tags: {ordered}")
            result.set_tags(ordered, section=self.section)
            log.info(f"Reordered {len(tags)} tags in section {self.section} using mode '{self.order_mode}'")
            return result

    def _order_by_category(self, tags: list[str], context: ImageContext) -> list[str]:
        """
        Order tags by semantic category using the tag cache.

        Rating and character tags are automatically pre‑categorized and excluded from
        the LLM call.
        """
        log.debug(f"[CLASSIFY] Starting classification for {len(tags)} tags")

        # Separate rating tags and character tags
        rating_tags = [t for t in tags if t.lower() in RATING_TAGS]
        character_tags_from_context = context.get_character_tags()
        # Also consider "original" and "borrowed_character" as special (we can treat as character)
        SPECIAL_CHARACTER = {"original", "borrowed_character"}
        special_tags = [t for t in tags if t.lower() in SPECIAL_CHARACTER]

        # Combine all tags that we want to pre‑categorize
        pre_categorized = set(rating_tags) | set(character_tags_from_context) | set(special_tags)

        # Tags that need LLM classification
        tags_to_classify = [t for t in tags if t not in pre_categorized]

        # Initialize cache
        cache = TagCategoryCache()

        # Manually set rating tags
        for tag in rating_tags:
            cache.set(tag, "rating")

        # Manually set character tags (including special ones)
        for tag in character_tags_from_context:
            cache.set(tag, "character name or series")
        for tag in special_tags:
            cache.set(tag, "character name or series")  # treat as character

        # Classify remaining tags via LLM
        if tags_to_classify:
            log.debug(f"[CLASSIFY] {len(tags_to_classify)} tags need LLM classification")
            try:
                classified = cache.classify_tags_batch(
                    tags_to_classify,
                    ollama_url=self.ollama_url,
                    model=self.model,
                    temperature=self.temperature,
                    timeout=self.timeout,
                    batch_size=self.batch_size,
                )
            except Exception as e:
                log.error(f"Classification failed: {e}")
                classified = {}
        else:
            classified = {}

        # Build final category mapping for all tags
        tag_cat = {}
        for tag in tags:
            # Use cache if available, else fallback to uncertain
            cached_cat = cache.get(tag)
            if cached_cat is not None:
                tag_cat[tag] = cached_cat
            elif tag in classified:
                tag_cat[tag] = classified[tag]
            else:
                tag_cat[tag] = "uncertain"

        log.debug(f"[CLASSIFY] Tag -> category mapping: {tag_cat}")

        # Sort by category order, then alphabetically within each category
        ordered = sorted(
            tags,
            key=lambda t: (CATEGORY_ORDER.index(tag_cat.get(t, "uncertain")), t)
        )
        log.debug(f"[CLASSIFY] Sorted order: {ordered}")
        return ordered

    def _order_rating_character(self, tags: list[str], context: ImageContext) -> list[str]:
        """
        Order as: rating → count tags → character tags → everything else.
        """
        rating = context.rating
        chars = context.get_character_tags()

        # Separate count tags
        count_tags = [t for t in tags if is_count_tag(t)]

        # The rest: tags that are not in chars, not rating, not count
        rest = [
            t for t in tags
            if t not in chars
            and (rating is None or t != rating)
            and not is_count_tag(t)
        ]

        ordered = []
        if rating:
            ordered.append(rating)
        ordered.extend(count_tags)   # count tags come after rating, before characters
        ordered.extend(chars)
        ordered.extend(rest)

        log.debug(f"[RATING_CHAR] Ordered: {ordered}")
        return ordered

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
from caption_pipeline.utils.tag_utils import sort_key_by_object
from caption_pipeline.utils.logging_utils import log, log_list_truncated, section
from caption_pipeline.utils.tag_cache import TagCategoryCache
from caption_pipeline.utils.tag_patterns import PRE_CATEGORIZE_PATTERNS

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

# Patterns for count/body tags (1boy, 2girls, multiple boys, etc.)
BODY_PATTERNS = [
    r"^(\d+)boys?$",  # 1boy, 2boys, 3boys
    r"^(\d+)girls?$",  # 1girl, 2girls, 3girls
    r"^(\d+)others?$",  # 1other, 2others
    r"^multiple\s+(boys|girls|others?)$",  # multiple boys, multiple girls
]


def is_count_tag(tag: str) -> bool:
    """Check if a tag is a count tag (e.g., 1girl, 2boys, solo, multiple boys)."""
    tag_lower = tag.lower().strip()
    if tag_lower == "solo":
        return True
    for pattern in BODY_PATTERNS:
        if re.match(pattern, tag_lower):
            return True
    return False


def is_body_tag(tag: str) -> bool:
    """Check if a tag should be classified as 'bodies' (count tags like 1girl, 2boys)."""
    return is_count_tag(tag)


@step_help(
    name="fix:order",
    description="Reorder tags in a specific section according to a mode.",
    long_description="""This step reorders tags in the given section.

Modes:
- category: Reorder by semantic category (using the tag cache). Tags are grouped by
  category and sorted alphabetically within each group. This is useful for consistent
  ordering across training examples. Unseen tags are classified via the LLM and
  added to the cache.

  Rating, body/count tags (1girl, 2boys, etc.), and character tags are automatically
  identified and pre‑categorized before the LLM call, so they are never sent to the LLM.
  Additionally, flexible regex patterns pre‑categorize many common tag suffixes
  (e.g., `*_hair` → body parts, `*_shirt` → wearables) to further reduce LLM usage.

- rating_character: Reorder as: rating tag (if any), then count tags, then character tags,
  then everything else (preserving original order for the rest). This matches the
  ordering used by format:join.

The step uses the persistent tag cache. If a tag is not yet in the cache, it will
be classified on the fly via the Ollama LLM.""",
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

            log.debug(f"[BEFORE] Original tags ({len(tags)}):")
            log_list_truncated(tags, "  Tags", max_items=10, level="debug")

            if self.order_mode == "category":
                ordered = self._order_by_category(tags, context)
            elif self.order_mode == "rating_character":
                ordered = self._order_rating_character(tags, context)
            else:
                raise ValueError(f"Unknown order_mode: {self.order_mode}")

            log.debug(f"[AFTER] Ordered tags ({len(ordered)}):")
            log_list_truncated(ordered, "  Tags", max_items=10, level="debug")

            result.set_tags(ordered, section=self.section)
            log.info(
                f"Reordered {len(tags)} tags in section {self.section} using mode '{self.order_mode}'"
            )
            return result

    def _order_by_category(self, tags: list[str], context: ImageContext) -> list[str]:
        """
        Order tags by semantic category using the tag cache.

        Rating, body/count tags (1girl, 2boys, etc.), character tags, and many
        common pattern-based tags are pre‑categorized and excluded from the LLM call.
        Within each category, tags are sorted by object (last word) for readability.
        """
        log.debug(f"[CLASSIFY] Starting classification for {len(tags)} tags")

        # Separate rating tags
        rating_tags = [t for t in tags if t.lower() in RATING_TAGS]

        # Separate body/count tags
        body_tags = [t for t in tags if is_body_tag(t)]

        # Separate character tags (from context) and special tags
        character_tags_from_context = context.get_character_tags()
        SPECIAL_CHARACTER = {"original", "borrowed_character"}
        special_tags = [t for t in tags if t.lower() in SPECIAL_CHARACTER]

        # Pattern-based pre-categorization
        pattern_categories = {}
        for tag in tags:
            # Skip tags already handled
            if (
                tag in rating_tags
                or tag in body_tags
                or tag in character_tags_from_context
                or tag in special_tags
            ):
                continue
            # Apply patterns
            for pattern, category in PRE_CATEGORIZE_PATTERNS:
                if pattern.match(tag):
                    pattern_categories[tag] = category
                    break

        # Log pre-categorization breakdown with truncation
        log.debug("[CLASSIFY] Pre-categorization breakdown:")
        log.debug(f"  Rating tags: {len(rating_tags)}")
        if rating_tags:
            log_list_truncated(rating_tags, "    Rating tags", max_items=5, level="debug")

        log.debug(f"  Body/count tags: {len(body_tags)}")
        if body_tags:
            log_list_truncated(body_tags, "    Body/count tags", max_items=5, level="debug")

        log.debug(f"  Character tags (from context): {len(character_tags_from_context)}")
        if character_tags_from_context:
            log_list_truncated(
                character_tags_from_context, "    Character tags", max_items=5, level="debug"
            )

        log.debug(f"  Special tags: {len(special_tags)}")
        if special_tags:
            log_list_truncated(special_tags, "    Special tags", max_items=5, level="debug")

        log.debug(f"  Pattern-categorized tags: {len(pattern_categories)}")
        if pattern_categories:
            # Group by category for better readability
            by_cat = {}
            for tag, cat in pattern_categories.items():
                by_cat.setdefault(cat, []).append(tag)
            for cat, tag_list in by_cat.items():
                log_list_truncated(tag_list, f"    {cat}", max_items=3, level="debug")

        # Combine all tags that we want to pre‑categorize
        pre_categorized = (
            set(rating_tags)
            | set(body_tags)
            | set(character_tags_from_context)
            | set(special_tags)
            | set(pattern_categories.keys())
        )

        # Tags that still need LLM classification
        tags_to_classify = [t for t in tags if t not in pre_categorized]
        log.debug(f"[CLASSIFY] Tags to classify via LLM: {len(tags_to_classify)}")
        if tags_to_classify:
            log_list_truncated(tags_to_classify, "  Tags to classify", max_items=5, level="debug")

        # Initialize cache
        cache = TagCategoryCache()

        # Manually set rating tags
        for tag in rating_tags:
            cache.set(tag, "rating")

        # Manually set body/count tags
        for tag in body_tags:
            cache.set(tag, "bodies")

        # Manually set character tags (including special ones)
        for tag in character_tags_from_context:
            cache.set(tag, "character name or series")
        for tag in special_tags:
            cache.set(tag, "character name or series")

        # Set pattern-based categories
        for tag, category in pattern_categories.items():
            cache.set(tag, category)

        # Classify remaining tags via LLM
        if tags_to_classify:
            log.debug(
                f"[CLASSIFY] {len(tags_to_classify)} tags need LLM classification (others pre-categorized)"
            )
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
            cached_cat = cache.get(tag)
            if cached_cat is not None:
                tag_cat[tag] = cached_cat
            elif tag in classified:
                tag_cat[tag] = classified[tag]
            else:
                tag_cat[tag] = "uncertain"

        # Log final mapping (truncated)
        log.debug("[CLASSIFY] Tag -> category mapping:")
        # Only show up to 10 entries, with more at continuation level
        for i, (tag, cat) in enumerate(sorted(tag_cat.items())):
            if i < 10:
                log.debug(f"    {tag} -> {cat}")
            elif i == 10:
                log.debug(f"    ... and {len(tag_cat) - 10} more")
                break

        # Sort by category order, then by object (last word) within each category
        ordered = sorted(
            tags,
            key=lambda t: (
                CATEGORY_ORDER.index(tag_cat.get(t, "uncertain")),
                sort_key_by_object(t),
            ),
        )
        log_list_truncated(ordered, "[CLASSIFY] Sorted order", max_items=10, level="debug")
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
            t
            for t in tags
            if t not in chars and (rating is None or t != rating) and not is_count_tag(t)
        ]

        ordered = []
        if rating:
            ordered.append(rating)
        ordered.extend(count_tags)
        ordered.extend(chars)
        ordered.extend(rest)

        log.debug(f"[RATING_CHAR] Ordered: {ordered}")
        return ordered

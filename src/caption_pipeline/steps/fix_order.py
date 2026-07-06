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

# src/caption_pipeline/steps/fix_order.py

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

@step_help(
    name="fix:order",
    description="Reorder tags in a specific section according to a mode.",
    long_description="""This step reorders tags in the given section.

Modes:
- category: Reorder by semantic category (using the tag cache). Tags are grouped by
  category and sorted alphabetically within each group. This is useful for consistent
  ordering across training examples. Unseen tags are classified via the LLM and
  added to the cache.
- rating_character: Reorder as: rating tag (if any), then character tags, then
  everything else (preserving original order). This matches the ordering used by
  format:join.

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
                ordered = self._order_by_category(tags)
            elif self.order_mode == "rating_character":
                ordered = self._order_rating_character(tags, context)
            else:
                raise ValueError(f"Unknown order_mode: {self.order_mode}")

            log.debug(f"[AFTER] Ordered tags: {ordered}")
            result.set_tags(ordered, section=self.section)
            log.info(f"Reordered {len(tags)} tags in section {self.section} using mode '{self.order_mode}'")
            return result

    def _order_by_category(self, tags: list[str]) -> list[str]:
        """Order tags by semantic category (using cache)."""
        log.debug(f"[CLASSIFY] Starting classification for {len(tags)} tags")
        cache = TagCategoryCache()
        try:
            classified = cache.classify_tags_batch(
                tags,
                ollama_url=self.ollama_url,
                model=self.model,
                temperature=self.temperature,
                timeout=self.timeout,
                batch_size=self.batch_size,
            )
        except Exception as e:
            log.error(f"Classification failed: {e}")
            classified = {}

        # Build mapping: use classified result if available, otherwise query the cache directly
        tag_cat = {}
        for tag in tags:
            if tag in classified:
                tag_cat[tag] = classified[tag]
            else:
                cached_cat = cache.get(tag)  # directly check cache
                tag_cat[tag] = cached_cat if cached_cat is not None else "uncertain"

        log.debug(f"[CLASSIFY] Tag -> category mapping: {tag_cat}")

        # Sort by category order, then alphabetically within each category
        ordered = sorted(
            tags,
            key=lambda t: (CATEGORY_ORDER.index(tag_cat.get(t, "uncertain")), t)
        )
        log.debug(f"[CLASSIFY] Sorted order: {ordered}")
        return ordered

    def _order_rating_character(self, tags: list[str], context: ImageContext) -> list[str]:
        """Order as: rating → characters → everything else."""
        rating = context.rating
        chars = context.character_tags
        # The rest: tags that are not in chars and not the rating (if rating exists)
        rest = [t for t in tags if t not in chars and (rating is None or t != rating)]
        ordered = []
        if rating:
            ordered.append(rating)
        ordered.extend(chars)
        ordered.extend(rest)
        log.debug(f"[RATING_CHAR] Ordered: {ordered}")
        return ordered

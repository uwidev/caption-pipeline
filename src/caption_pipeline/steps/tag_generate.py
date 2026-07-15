# src/caption_pipeline/steps/tag_generate.py

from pathlib import Path
from typing import Any, ClassVar

import torch
import vibe
from PIL import Image
from vibe.result_processors import CleanTags, ScoreThresholds, TagLevelThresholds

from caption_pipeline.core.context import ImageContext
from caption_pipeline.core.help import step_help
from caption_pipeline.core.step import PipelineStep
from caption_pipeline.utils.logging_utils import (
    log,
    log_list_truncated,
    log_scored_list_truncated,
    section,
)
from caption_pipeline.utils.tag_db import (
    get_character_count_from_tag_confidences,
    load_tag_databases,
    resolve_character_tags,
)
from caption_pipeline.utils.tokenizer import get_tokenizer

ALWAYS_BLACKLIST: set[str] = {
    "virtual youtuber",
    "dual persona",
    "ranguage",
}

# Constants for user tag penalty
USER_TAG_PENALTY_MAX = 0.90
USER_TAG_PENALTY_MIN = 0.55
USER_TAG_SATURATION = 15


@step_help(
    name="tag:generate",
    description="Generate AI tags from images using vibe (AnimeTimm CaFormer).",
    long_description="""This step runs AI inference using the vibe library with an AnimeTimm model
(default: at-caformer-b36-dbv4-full) to generate Danbooru-style tags.
It merges user-provided hints with AI results, detects character tags, and applies filters.

IMPORTANT: If the user provides character tags in the grounding/hints, those
take precedence over AI-inferenced character tags. The step will NOT use
AI-inferenced characters if any user-provided characters exist.

The model can be changed via --model and a local folder via --source.""",
    options=[
        {"flag": "--threshold FLOAT", "help": "Confidence threshold for tags", "default": "0.35"},
        {
            "flag": "--character-threshold FLOAT",
            "help": "Threshold for character tags",
            "default": "0.75",
        },
        {"flag": "--whitelist TAG,TAG,...", "help": "Tags to always keep (overrides all filters)"},
        {"flag": "--blacklist TAG,TAG,...", "help": "Tags to always remove"},
        {
            "flag": "--model ID",
            "help": "Vibe model ID (e.g., at-caformer-b36-dbv4-full)",
            "default": "at-caformer-b36-dbv4-full",
        },
        {"flag": "--source PATH", "help": "Local folder or HF repo for model files"},
        {"flag": "--no-infer-characters", "help": "Don't infer character names from AI"},
        {
            "flag": "--no-unload-models",
            "help": "Keep models loaded after batch (faster but uses more VRAM)",
            "default": "unloaded",
        },
        {"flag": "--no-use-hints", "help": "Ignore user-provided tags", "default": "use hints"},
    ],
    example="tag:generate --threshold 0.35 --model at-caformer-b36-dbv4-full --source local:/path/to/model",
)
class TagGenerationStep(PipelineStep):
    """
    Generate tags from images using vibe with AnimeTimm models.

    Character handling priority:
    1. User-provided character tags (from grounding/hints) → ALWAYS used
    2. AI-inferenced character tags → ONLY used if no user characters exist
    """

    # Class-level caches for shared resources
    _general_tags: ClassVar[set[str] | None] = None
    _character_tags: ClassVar[set[str] | None] = None
    _tokenizer: ClassVar[Any] = None
    _vibe_session: ClassVar[Any] = None  # shared session across batch
    _session_refcount: ClassVar[int] = 0

    def __init__(
        self,
        threshold: float = 0.35,
        character_threshold: float = 0.75,
        whitelist: list[str] | None = None,
        blacklist: list[str] | None = None,
        danbooru_only: bool = False,
        use_user_hints: bool = True,
        user_bonus: float = 1.0,
        ai_penalty: float = 0.66,
        infer_characters: bool = False,
        unload_models_after_batch: bool = True,
        user_tag_penalty_min: float = USER_TAG_PENALTY_MIN,
        user_tag_penalty_max: float = USER_TAG_PENALTY_MAX,
        user_tag_saturation: int = USER_TAG_SATURATION,
        model_id: str = "at-caformer-b36-dbv4-full",
        model_source: str | None = None,
        use_tag_level_thresholds: bool = True,
        tag_level_threshold_offset: float = 0.0,
        tag_level_threshold_fallback: float | None = None,
    ) -> None:
        self.threshold: float = threshold
        self.character_threshold: float = character_threshold
        self.whitelist: set[str] = set(whitelist or [])
        self.blacklist: set[str] = set(blacklist or [])
        self.danbooru_only: bool = danbooru_only
        self.use_user_hints: bool = use_user_hints
        self.user_bonus: float = user_bonus
        self.ai_penalty: float = ai_penalty
        self.infer_characters: bool = infer_characters
        self.unload_models_after_batch: bool = unload_models_after_batch
        self.user_tag_penalty_min: float = user_tag_penalty_min
        self.user_tag_penalty_max: float = user_tag_penalty_max
        self.user_tag_saturation: int = user_tag_saturation
        self.use_tag_level_thresholds: bool = use_tag_level_thresholds
        self.tag_level_threshold_offset: float = tag_level_threshold_offset
        self.tag_level_threshold_fallback: float | None = tag_level_threshold_fallback

        # Vibe model configuration
        self.model_id: str = model_id
        self.model_source: str | None = model_source
        self._session: Any = None  # per‑batch session, if not using class cache

    def name(self) -> str:
        return "tag:generate"

    def validate(self, context: ImageContext) -> bool:
        return context.image_path.exists()

    def process(self, context: ImageContext) -> ImageContext | None:
        """Generate tags for the image."""
        with section(f"Processing: {context.image_path.name}"):
            self._load_databases()
            self._ensure_vibe_session()

            image: Image.Image = context.load_image()

            # Run AI inference via vibe
            ai_tags, ai_rating, ai_characters = self._run_vibe_inference(image)

            user_tags = context.get_tags(1)
            user_characters = context.get_character_tags()
            user_rating = context.rating

            # Use character_threshold for filtering AI characters
            accepted_ai_characters = [
                character
                for character, score in ai_characters.items()
                if score > self.character_threshold
            ]

            if user_tags:
                log_list_truncated(user_tags, "User tags", level="debug")

            if user_rating:
                log.debug(f"User rating: {user_rating}")

            log.debug(f"AI inference results ({len(ai_tags)} total tags):")
            sorted_ai = sorted(ai_tags.items(), key=lambda x: -x[1])

            above_threshold = [(tag, conf) for tag, conf in sorted_ai if conf >= self.threshold]
            below_threshold = [(tag, conf) for tag, conf in sorted_ai if conf < self.threshold]

            if above_threshold:
                log_scored_list_truncated(above_threshold, "Tags above threshold")
            else:
                log.debug(f"No tags above threshold ({self.threshold})")

            if below_threshold:
                near_threshold = [
                    (tag, conf) for tag, conf in below_threshold if conf >= self.threshold - 0.1
                ]
                if near_threshold:
                    log_scored_list_truncated(near_threshold[:10], "Tags near threshold")
                    if len(near_threshold) > 10:
                        log.debug(f"... and {len(near_threshold) - 10} more near threshold")
                else:
                    highest_below = below_threshold[:5]
                    log_scored_list_truncated(highest_below, "Highest below threshold")

            if ai_rating:
                log.debug(f"AI rating: {ai_rating}")

            if ai_characters:
                sorted_chars = sorted(ai_characters.items(), key=lambda x: -x[1])
                formatted = [
                    f"{char}: {conf:.3f} {'✓' if conf >= self.character_threshold else ' '}"
                    for char, conf in sorted_chars
                ]
                log_list_truncated(
                    formatted, "AI-inferenced characters", max_items=10, level="debug"
                )

            combined_general = self._combine_tags(
                user_tags=user_tags,
                ai_tags=ai_tags,
            )

            expected_count = get_character_count_from_tag_confidences(combined_general)

            if self.infer_characters and expected_count > 0:
                user_count = len(user_characters)
                ai_count = len(accepted_ai_characters)
                total_available = user_count + ai_count

                if total_available < expected_count:
                    needed = expected_count - total_available
                    log.warning(
                        f"Not enough character tags for {context.image_path.name}: "
                        f"expected {expected_count} characters from count tags, "
                        f"but only {user_count} user + {ai_count} AI = {total_available} available. "
                        f"Missing {needed} character(s). Try lowering --character-threshold or adding more character hints."
                    )

            resolved_characters = resolve_character_tags(
                user_character_tags=user_characters,
                ai_character_tags=accepted_ai_characters,
                count=expected_count,
                allow_ai=self.infer_characters,
                all_tags=list(combined_general.keys()),
                context_name=context.image_path.name,
                threshold=self.character_threshold,
            )

            final_tags = self._apply_filters(combined_general)

            # --- Deltas logging ---
            original_tag_count = len(user_tags)
            final_tag_count = len(final_tags)

            user_tag_set = set(user_tags)
            added_by_ai = [t for t in final_tags if t not in user_tag_set]
            removed = [t for t in user_tags if t not in final_tags]
            kept = [t for t in final_tags if t in user_tag_set]

            final_char_tags = resolved_characters
            user_char_set = set(user_characters)

            kept_chars = [c for c in final_char_tags if c in user_char_set]
            added_chars = [c for c in final_char_tags if c not in user_char_set]

            log.info(f"{original_tag_count} user tags → {final_tag_count} final tags")

            if user_rating and ai_rating:
                if user_rating == ai_rating:
                    log.info(f"Rating: {user_rating} (user and AI match)")
                else:
                    log.info(
                        f"Rating: {user_rating} (user) vs {ai_rating} (AI) - using user rating"
                    )
            elif user_rating:
                log.info(f"Rating: {user_rating} (user provided)")
            elif ai_rating:
                log.info(f"Rating: {ai_rating} (AI inferred)")

            if kept:
                log_list_truncated(kept, "Kept")
            if added_by_ai:
                log_list_truncated(added_by_ai, "Added by AI")
            if removed:
                log_list_truncated(removed, "Removed by AI")
            if kept_chars:
                log_list_truncated(kept_chars, "Kept characters")
            if added_chars:
                log_list_truncated(added_chars, "Characters added")
            if final_tags:
                log_list_truncated(final_tags, "Final tags")

            # Build result
            result = context.copy()
            result.inferenced_tags = ai_tags
            result.set_tags(list(final_tags), section=1)
            result.set_characters(resolved_characters)

            if user_rating:
                result.rating = user_rating
            elif ai_rating:
                result.rating = ai_rating
            else:
                result.rating = None

            return result

    def process_batch(self, contexts: list[ImageContext]) -> list[ImageContext]:
        """Process multiple contexts with models loaded once."""
        if not contexts:
            return contexts

        self._load_databases()
        self._ensure_vibe_session()

        valid_indices: list[int] = []
        for idx, context in enumerate(contexts):
            if self.validate(context):
                try:
                    context.load_image()
                    valid_indices.append(idx)
                except Exception as e:
                    log.error(f"Failed to load image {context.image_path.name}: {e}")

        if not valid_indices:
            return contexts

        results: list[tuple[int, ImageContext]] = []

        for idx in valid_indices:
            context: ImageContext = contexts[idx]
            try:
                result: ImageContext | None = self.process(context)
                if result is not None:
                    results.append((idx, result))
                else:
                    results.append((idx, context))
            except Exception as e:
                log.error(f"Failed to process {context.image_path.name}: {e}")
                results.append((idx, context))

        results.sort(key=lambda x: x[0])
        processed_contexts: list[ImageContext] = [r[1] for r in results]

        for pos, idx in enumerate(valid_indices):
            contexts[idx] = processed_contexts[pos]

        if self.unload_models_after_batch:
            self._unload_vibe_session()

        return contexts

    # =========================================================================
    # Vibe Session Management (replaces imgutils model loading)
    # =========================================================================

    def _ensure_vibe_session(self) -> None:
        """Load the vibe session if not already loaded."""
        if self._session is not None:
            return
        self._load_vibe_session()

    def _load_vibe_session(self) -> None:
        """Load vibe session for this batch."""
        if self._session is not None:
            return

        device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info(f"Loading vibe model '{self.model_id}' on {device}...")

        try:
            self._session = vibe.load(
                self.model_id,
                source=self.model_source,
                device=device,
                precision="fp16" if device == "cuda" else "fp32",
                auto_download=True,
            )
            log.info(f"Vibe model '{self.model_id}' loaded successfully.")
        except Exception as e:
            log.error(f"Failed to load vibe model: {e}")
            raise

    def _unload_vibe_session(self) -> None:
        """Unload vibe session and clean up."""
        if self._session is not None:
            self._session.close()
            self._session = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            log.debug("Vibe session closed and CUDA cache cleared.")

    # =========================================================================
    # AI Inference using Vibe
    # =========================================================================

    def _run_vibe_inference(
        self,
        image: Image.Image,
        context_name: str | None = None,
    ) -> tuple[dict[str, float], str | None, dict[str, float]]:
        """Run inference using vibe with Tag-Level Thresholds."""
        if self._session is None:
            raise RuntimeError("Vibe session not loaded.")

        # --- Step 1: Run inference once ---
        try:
            raw_result = self._session.infer(image).first()
        except Exception as e:
            log.error(f"Vibe inference failed: {e}")
            return {}, None, {}

        # --- Step 2: Get threshold map for the model ---
        threshold_map = self._get_threshold_map()

        # --- Step 3: Log raw scores with thresholds ---
        context_label = f" for {context_name}" if context_name else ""

        # Collect and sort all tags by score
        all_tags = []
        for category, entries in raw_result.tags.items():
            for entry in entries:
                all_tags.append((category, entry.tag, entry.score))

        all_tags.sort(key=lambda x: x[2], reverse=True)

        # Format tags with alternating dot padding for visual readability
        formatted_tags = []
        for idx, (_, tag, score) in enumerate(all_tags[:50]):  # Only format top 50
            threshold = threshold_map.get(tag)

            # Alternate: even indices (0, 2, 4...) get dots, odd indices (1, 3, 5...) get spaces
            if idx % 2 == 0:
                padded_tag = tag.ljust(40, ".")
            else:
                padded_tag = f"{tag:<40}"

            if threshold is not None:
                status = "✓" if score >= threshold else "✗"
                formatted_tags.append(
                    f"{padded_tag}score={score:.3f} | threshold={threshold:.3f} [{status}]"
                )
            else:
                formatted_tags.append(f"{padded_tag}score={score:.3f} | threshold=N/A [?]")

        # Log with truncation - show up to 10 tags
        log_list_truncated(
            items=formatted_tags,
            message=f"Raw tags with thresholds{context_label}",
            max_items=10,
        )

        total_raw = len(all_tags)

        if total_raw: # debug print for previous log_list_truncated to indicate more
            log.debug('     ...')

        # --- Step 4: Apply processors (TLT or global threshold + CleanTags) ---
        processors = []

        if self.use_tag_level_thresholds:
            tlt_kwargs = {}
            if self.tag_level_threshold_offset != 0.0:
                tlt_kwargs["threshold_relative_offset"] = self.tag_level_threshold_offset
            if self.tag_level_threshold_fallback is not None:
                tlt_kwargs["threshold_fallback"] = self.tag_level_threshold_fallback
            processors.append(TagLevelThresholds(**tlt_kwargs))
        else:
            processors.append(ScoreThresholds(threshold=self.threshold))

        processors.append(CleanTags())

        try:
            filtered_result = self._session.infer(image, result_processors=processors).first()
        except Exception as e:
            log.error(f"Failed to apply thresholds: {e}")
            return {}, None, {}

        # --- Step 5: Log summary of filtering ---
        total_filtered = sum(len(entries) for entries in filtered_result.tags.values())
        dropped = total_raw - total_filtered

        log.debug(
            f"Threshold summary{context_label}: "
            f"{total_raw} raw tags → {total_filtered} kept ({dropped} dropped)"
        )

        # --- Step 6: Extract results ---
        rating_entries = filtered_result.tags.get("rating", [])
        rating = None
        if rating_entries:
            best = max(rating_entries, key=lambda e: e.score)
            rating = best.tag

        general_tags = {entry.tag: entry.score for entry in filtered_result.tags.get("general", [])}
        character_tags = {
            entry.tag: entry.score for entry in filtered_result.tags.get("character", [])
        }

        return general_tags, rating, character_tags

    def _find_selected_tags_csv(self) -> Path | None:
        """Find the selected_tags.csv file for the current model."""
        if self._session is None:
            return None

        # The session stores the file_map
        if hasattr(self._session, "_file_map"):
            file_map = self._session._file_map
            if hasattr(file_map, "as_path_dict"):
                paths = file_map.as_path_dict()
                for key, path in paths.items():
                    if "selected_tags" in key or (isinstance(key, str) and key.endswith(".csv")):
                        return path
        return None

    def _get_threshold_map(self) -> dict[str, float]:
        """Extract the threshold map from the vibe session or cache."""
        if not self.use_tag_level_thresholds:
            return {}

        # Check if we already cached the threshold map
        if hasattr(self, "_threshold_map_cache"):
            return self._threshold_map_cache

        # Try to get it from the session's file_map
        threshold_map = {}
        if self._session is not None:
            # Access the file_map from the session
            if hasattr(self._session, "_file_map"):
                file_map = self._session._file_map
                if hasattr(file_map, "as_path_dict"):
                    paths = file_map.as_path_dict()
                    for key, path in paths.items():
                        if "selected_tags" in key or (
                            isinstance(key, str) and key.endswith(".csv")
                        ):
                            threshold_map = self._parse_threshold_csv(path)
                            break

        # Cache it for future use
        self._threshold_map_cache = threshold_map
        return threshold_map

    def _parse_threshold_csv(self, csv_path: Path) -> dict[str, float]:
        """Parse selected_tags.csv and extract tag->threshold mapping."""
        import csv

        threshold_map = {}
        try:
            with csv_path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    tag = row.get("name", "").strip()
                    threshold_str = row.get("best_threshold", "").strip()
                    if tag and threshold_str:
                        try:
                            threshold_map[tag] = float(threshold_str)
                        except ValueError:
                            pass
        except Exception as e:
            log.warning(f"Failed to parse threshold CSV: {e}")

        return threshold_map

    # =========================================================================
    # Database Loading
    # =========================================================================

    @classmethod
    def _load_databases(cls) -> None:
        if cls._general_tags is not None:
            return

        general_tags, character_tags = load_tag_databases()
        cls._general_tags = set(general_tags)
        cls._character_tags = set(character_tags)

        if cls._tokenizer is None:
            cls._tokenizer = get_tokenizer()

    # =========================================================================
    # Tag Combination and Filtering
    # =========================================================================

    def _combine_tags(
        self,
        user_tags: list[str],
        ai_tags: dict[str, float],
    ) -> dict[str, float]:
        """
        Combine user and AI tags.

        AI tags are already filtered by TLT (or global threshold) in _run_vibe_inference.
        """
        combined_general: dict[str, float] = {}

        # Start with AI tags (already filtered)
        for tag, conf in ai_tags.items():
            combined_general[tag] = conf

        # Apply user tag penalty/boost if using hints
        if user_tags and self.use_user_hints:
            user_count = len(user_tags)
            penalty = self.user_tag_penalty_max - (
                self.user_tag_penalty_max - self.user_tag_penalty_min
            ) * min(user_count / self.user_tag_saturation, 1.0)

            # Penalize AI tags not in user tags
            for tag in list(combined_general.keys()):
                if tag not in user_tags:
                    combined_general[tag] = combined_general[tag] * penalty

            # Boost user tags
            for tag in user_tags:
                if tag in combined_general:
                    combined_general[tag] = min(1.0, combined_general[tag] / penalty)
                    combined_general[tag] = min(1.0, combined_general[tag] + self.user_bonus)
                else:
                    combined_general[tag] = 0.95

        return combined_general

    def _apply_filters(self, tags: dict[str, float]) -> list[str]:
        """Apply blacklist/whitelist/danbooru_only."""
        result: list[str] = []

        for tag, conf in tags.items():
            if tag in self.blacklist or tag in ALWAYS_BLACKLIST:
                continue

            if self.danbooru_only:
                if self._general_tags and tag not in self._general_tags:
                    if tag not in self.whitelist:
                        continue

            result.append(tag)

        result.sort(key=lambda x: tags.get(x, 0), reverse=True)
        return result

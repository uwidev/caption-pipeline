"""
TagGenerationStep: Generate tags using AI models with user hints.
"""

import gc
import re
from typing import Any, ClassVar

import torch
from PIL import Image
from transformers import PreTrainedTokenizerBase

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
}

# Constants (adjustable)
USER_TAG_PENALTY_MAX = 0.90  # Minimum penalty (least aggressive)
USER_TAG_PENALTY_MIN = 0.55  # Maximum penalty (most aggressive)
USER_TAG_SATURATION = 15  # Number of user tags before max penalty is applied


@step_help(
    name="tag:generate",
    description="Generate AI tags from images using WD14 and PixAI.",
    long_description="""This step runs AI inference using WD14 (EVA02_Large) and PixAI (v0.9)
models to generate danbooru-style tags. It merges user-provided hints with AI
results, detects character tags, and applies filters.

IMPORTANT: If the user provides character tags in the grounding/hints, those
take precedence over AI-inferenced character tags. The step will NOT use
AI-inferenced characters if any user-provided characters exist.""",
    options=[
        {"flag": "--threshold FLOAT", "help": "Confidence threshold for tags", "default": "0.35"},
        {"flag": "--whitelist TAG,TAG,...", "help": "Tags to always keep (overrides all filters)"},
        {"flag": "--blacklist TAG,TAG,...", "help": "Tags to always remove"},
        {"flag": "--no-infer-characters", "help": "Don't infer character names from AI"},
        {
            "flag": "--no-unload-models",
            "help": "Keep models loaded after batch (faster but uses more VRAM)",
            "default": "unloaded",
        },
        {"flag": "--no-use-hints", "help": "Ignore user-provided tags", "default": "use hints"},
    ],
    example="tag:generate --threshold 0.35 --whitelist '1girl, original' --no-infer-characters",
)
class TagGenerationStep(PipelineStep):
    """
    Generate tags from images using AI models with user hints.

    Character handling priority:
    1. User-provided character tags (from grounding/hints) → ALWAYS used
    2. AI-inferenced character tags → ONLY used if no user characters exist
    """

    # Class-level caches for shared resources
    _general_tags: ClassVar[set[str] | None] = None
    _character_tags: ClassVar[set[str] | None] = None
    _tokenizer: ClassVar[PreTrainedTokenizerBase | None] = None
    _models_loaded: ClassVar[bool] = False
    _model_instances: ClassVar[dict[str, Any]] = {}

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

    def name(self) -> str:
        return "tag:generate"

    def validate(self, context: ImageContext) -> bool:
        return context.image_path.exists()

    def process(self, context: ImageContext) -> ImageContext | None:
        """Generate tags for the image."""
        with section(f"Processing: {context.image_path.name}"):
            self._load_databases()
            self._load_models()

            image: Image.Image = context.load_image()

            # Run AI inference
            ai_tags, ai_rating, ai_characters = self._run_inference(image)

            user_tags = context.get_tags(0) + context.get_tags(1)
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

            # Log user rating at DEBUG level
            if user_rating:
                log.debug(f"User rating: {user_rating}")

            # Log AI inference results at DEBUG level
            log.debug(f"AI inference results ({len(ai_tags)} total tags):")
            sorted_ai = sorted(ai_tags.items(), key=lambda x: -x[1])

            # Show ALL tags above threshold
            above_threshold = [(tag, conf) for tag, conf in sorted_ai if conf >= self.threshold]
            below_threshold = [(tag, conf) for tag, conf in sorted_ai if conf < self.threshold]

            if above_threshold:
                log_scored_list_truncated(above_threshold, "Tags above threshold")
            else:
                log.debug(f"No tags above threshold ({self.threshold})")

            # Show tags near threshold (within 0.1) + up to 10 more
            if below_threshold:
                near_threshold = [
                    (tag, conf) for tag, conf in below_threshold if conf >= self.threshold - 0.1
                ]
                if near_threshold:
                    log_scored_list_truncated(near_threshold[:10], "Tags near threshold")
                    if len(near_threshold) > 10:
                        log.debug(f"... and {len(near_threshold) - 10} more near threshold")

                else:
                    # If no tags near threshold, show the highest below threshold
                    highest_below = below_threshold[:5]
                    log_scored_list_truncated(highest_below, "Highest below threshold")

            # Log AI rating at DEBUG level
            if ai_rating:
                log.debug(f"AI rating: {ai_rating}")

            # Log ALL AI character results at DEBUG level (no filtering)
            if ai_characters:
                sorted_chars = sorted(ai_characters.items(), key=lambda x: -x[1])
                # Format with markers
                formatted = [
                    f"{char}: {conf:.3f} {'✓' if conf >= self.character_threshold else ' '}"
                    for char, conf in sorted_chars
                ]
                log_list_truncated(
                    formatted, "AI-inferenced characters", max_items=10, level="debug"
                )

            # Combine general tags (using regular threshold)
            combined_general = self._combine_tags(
                user_tags=user_tags,
                ai_tags=ai_tags,
            )

            expected_count = get_character_count_from_tag_confidences(combined_general)

            # Check if we have enough AI characters (if AI character detection is allowed)
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

            # Apply filters (blacklist/whitelist/danbooru_only)
            final_tags = self._apply_filters(combined_general)

            # === Show deltas: What changed ===
            original_tag_count = len(user_tags)
            final_tag_count = len(final_tags)

            # Tags that were added by AI (in final but not in user hints)
            user_tag_set = set(user_tags)
            added_by_ai = [t for t in final_tags if t not in user_tag_set]

            # Tags that were removed (in user hints but not in final)
            removed = [t for t in user_tags if t not in final_tags]

            # Tags that were kept (in both user hints and final)
            kept = [t for t in final_tags if t in user_tag_set]

            # Characters added/kept
            final_char_tags = resolved_characters
            user_char_set = set(user_characters)

            kept_chars = [c for c in final_char_tags if c in user_char_set]
            added_chars = [c for c in final_char_tags if c not in user_char_set]

            # Show summary at INFO level
            log.info(f"{original_tag_count} user tags → {final_tag_count} final tags")

            # Log rating delta if applicable
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

            # Determine final rating: user rating takes precedence
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
        self._load_models()

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
            self._unload_models()

        return contexts

    # =========================================================================
    # Model Loading
    # =========================================================================

    @classmethod
    def _load_models(cls) -> None:
        if cls._models_loaded:
            return

        log.info("Loading tag generation models...")

        try:
            # This import triggers model loading
            from imgutils.generic.classify import ClassifyModel
            from imgutils.generic.yolo import YOLOModel
            from imgutils.tagging import pixai, wd14

            # Force model loading by creating dummy prediction
            wd_model = ClassifyModel("wd14")
            cls._model_instances["wd14"] = wd_model

            pixai_model = YOLOModel("pixai")
            cls._model_instances["pixai"] = pixai_model

            # Create a dummy image to trigger model loading
            dummy = Image.new("RGB", (224, 224), color="white")

            # Load WD14 model
            wd14.get_wd14_tags(
                dummy,
                "EVA02_Large",
                no_underline=True,
                general_threshold=0.1,
            )

            # Load PixAI model
            pixai.get_pixai_tags(
                dummy,
                "v0.9",
                thresholds={"general": 0.1, "character": 0.1},
            )

            cls._models_loaded = True
            log.info("Tag generation models loaded")

        except Exception as e:
            log.warning(f"Failed to load models: {e}")
            # Mark as loaded anyway to prevent repeated attempts
            cls._models_loaded = True

    @classmethod
    def _unload_models(cls) -> None:
        """Properly unload all imgutils models and clear caches."""
        if not cls._models_loaded:
            return

        log.info("Unloading tag generation models...")

        try:
            # 1. Clear model instances from our internal cache
            for model_name, model_instance in cls._model_instances.items():
                if hasattr(model_instance, "clear"):
                    try:
                        model_instance.clear()
                    except Exception as e:
                        log.warning(f"Failed to clear {model_name}: {e}")

            # 2. Clear all imgutils model caches
            cls._clear_imgutils_caches()

            # 3. Force garbage collection
            gc.collect()

            # 4. Clear CUDA cache if available
            if torch.cuda.is_available():
                # Synchronize first
                torch.cuda.synchronize()

                # Empty cache
                torch.cuda.empty_cache()

                # Reset peak memory stats
                try:
                    torch.cuda.reset_peak_memory_stats()
                except Exception:
                    pass

                # Log memory state at DEBUG level
                allocated = torch.cuda.memory_allocated() / 1024 / 1024
                cached = torch.cuda.memory_reserved() / 1024 / 1024
                if allocated > 0 or cached > 0:
                    log.debug(
                        f"CUDA memory after cleanup: {allocated:.2f}MB allocated, {cached:.2f}MB cached"
                    )
                else:
                    log.debug("CUDA memory fully released")

                # Force another GC pass after CUDA cleanup
                gc.collect()

            # 5. Clear our internal state
            cls._model_instances.clear()
            cls._models_loaded = False

            log.info("Tag generation models unloaded")

        except Exception as e:
            log.warning(f"Failed to unload models: {e}")
            cls._models_loaded = False

    @classmethod
    def _clear_imgutils_caches(cls) -> None:
        """Clear all imgutils cached functions."""
        import inspect
        import sys

        cleared = 0

        # Functions that we know are cached and used by our models
        known_cached_funcs = [
            # WD14
            ("imgutils.tagging.wd14", "_get_wd14_model"),
            ("imgutils.tagging.wd14", "_get_wd14_weights"),
            ("imgutils.tagging.wd14", "_get_wd14_labels"),
            # PixAI
            ("imgutils.tagging.pixai", "_open_onnx_model"),
            ("imgutils.tagging.pixai", "_open_tags"),
            ("imgutils.tagging.pixai", "_open_preprocess"),
            ("imgutils.tagging.pixai", "_open_default_category_thresholds"),
            # YOLO
            ("imgutils.generic.yolo", "_open_models_for_repo_id"),
            # Classify
            ("imgutils.generic.classify", "_open_models_for_repo_id"),
            # Booru YOLO
            ("imgutils.detect.booru_yolo", "_open_models_for_repo_id"),
            # Person detect
            ("imgutils.detect.person", "_open_models_for_repo_id"),
        ]

        for module_name, func_name in known_cached_funcs:
            try:
                # Import the module
                __import__(module_name)
                module = sys.modules[module_name]

                # Get the function
                if hasattr(module, func_name):
                    func = getattr(module, func_name)
                    if hasattr(func, "cache_clear") and callable(func.cache_clear):
                        func.cache_clear()
                        cleared += 1
            except Exception:
                pass

        # Also try to find any other cached functions in imgutils modules
        for module_name, module in list(sys.modules.items()):
            if not module_name.startswith("imgutils"):
                continue

            for name, obj in inspect.getmembers(module):
                # Check if this is a cached function
                if hasattr(obj, "cache_clear") and callable(obj.cache_clear):
                    try:
                        # Check if it was decorated with ts_lru_cache
                        # The decorator adds a __wrapped__ attribute
                        if hasattr(obj, "__wrapped__"):
                            obj.cache_clear()
                            cleared += 1
                    except Exception:
                        pass

        log.debug(f"Cleared {cleared} imgutils caches")

    # =========================================================================
    # Database Loading
    # =========================================================================

    @classmethod
    def _load_databases(cls) -> None:
        if cls._general_tags is not None:
            return

        general_tags: list[str]
        character_tags: list[str]
        general_tags, character_tags = load_tag_databases()
        cls._general_tags = set(general_tags)
        cls._character_tags = set(character_tags)

        if cls._tokenizer is None:
            cls._tokenizer = get_tokenizer()

    # =========================================================================
    # AI Inference
    # =========================================================================

    def _run_inference(
        self,
        image: Image.Image,
    ) -> tuple[dict[str, float], str | None, dict[str, float]]:
        from imgutils.tagging import pixai, wd14

        low_threshold = 0.01

        # WD14 returns: (ratings_dict, general_tags_dict, character_tags_dict)
        wd_ratings: dict[str, float]
        wd_general: dict[str, float]
        wd_ratings, wd_general, _ = wd14.get_wd14_tags(
            image,
            "EVA02_Large",
            no_underline=True,
            general_threshold=low_threshold,
        )

        # Extract the highest confidence rating from the ratings dict
        rating: str | None = None
        if wd_ratings:
            rating_order = ["safe", "questionable", "explicit", "general", "sensitive"]
            best_rating = None
            best_conf = -1.0
            for r in rating_order:
                conf = wd_ratings.get(r, 0.0)
                if conf > best_conf:
                    best_rating = r
                    best_conf = conf
            rating = best_rating

            # log.debug(f"Extracted rating: {rating} (from {wd_ratings})")

        pixai_general: dict[str, float]
        pixai_characters: dict[str, float]
        pixai_general, pixai_characters = pixai.get_pixai_tags(
            image,
            "v0.9",
            thresholds={
                "general": low_threshold,
                "character": 0.1,
            },
        )

        pixai_general = {self._normalize_tag(k): v for k, v in pixai_general.items()}
        pixai_characters = {self._normalize_tag(k): v for k, v in pixai_characters.items()}

        combined: dict[str, float] = {}
        for tag, conf in wd_general.items():
            combined[tag] = conf

        for tag, conf in pixai_general.items():
            combined[tag] = max(combined.get(tag, 0), conf)

        # If WD14 didn't give a rating, try PixAI
        if not rating and pixai_general:
            for tag in ["general", "safe", "questionable", "explicit"]:
                if tag in pixai_general:
                    rating = tag
                    break

        return combined, rating, pixai_characters

    def _normalize_tag(self, tag: str) -> str:
        return tag.replace("_", " ")

    # =========================================================================
    # Tag Combination
    # =========================================================================

    def _combine_tags(
        self,
        user_tags: list[str],
        ai_tags: dict[str, float],
    ) -> dict[str, float]:
        """
        Combine user tags and AI tags into a single set above threshold.

        Args:
            user_tags: User-provided tags (already normalized)
            ai_tags: AI-inferenced general tags

        Returns:
            Tuple of combined_general_tags
        """
        combined_general: dict[str, float] = {}

        # Start with AI general tags
        for tag, conf in ai_tags.items():
            combined_general[tag] = conf

        # Apply user tag penalty/boost
        if user_tags:
            user_count = len(user_tags)

            # Calculate penalty based on user tag count
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
                    # Boost existing AI tag
                    combined_general[tag] = min(1.0, combined_general[tag] / penalty)
                    combined_general[tag] = min(1.0, combined_general[tag] + self.user_bonus)
                else:
                    # Add user tag if missing
                    combined_general[tag] = 0.95

        # Filter by threshold
        final_general = {
            tag: conf for tag, conf in combined_general.items() if conf >= self.threshold
        }

        return final_general

    # =========================================================================
    # Filtering
    # =========================================================================

    def _apply_filters(self, tags: dict[str, float]) -> list[str]:
        """Filter against blacklist or non-danbooru tags."""
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

    def _clean_tag(self, tag: str) -> str:
        if not tag:
            return ""

        cleaned = tag.strip()
        cleaned = " ".join(cleaned.split())
        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = cleaned.strip(".,;:!?\"'")

        if not cleaned:
            return ""

        return cleaned

"""
TagGenerationStep: Generate tags using AI models with user hints.
"""

import ast
import gc
import os
import re
import time
from pathlib import Path
from typing import Any, ClassVar

import torch
from loguru import logger
from PIL import Image
from transformers import PreTrainedTokenizerBase

from caption_pipeline.core.context import ImageContext
from caption_pipeline.core.step import PipelineStep
from caption_pipeline.core.help import step_help
from caption_pipeline.utils.booru_characters import DanbooruCharacters
from caption_pipeline.utils.character_extractor import (
    CharacterEntry,
    CharacterExtractor,
    CharacterSource,
    get_character_database,
)
from caption_pipeline.utils.tag_db import load_tag_databases
from caption_pipeline.utils.tokenizer import get_tokenizer


ALWAYS_BLACKLIST: set[str] = {
    "virtual youtuber",
    "borrowed character",
    "dual persona",
}

# Constants (adjustable)
USER_TAG_PENALTY_MAX = 0.90  # Minimum penalty (least aggressive)
USER_TAG_PENALTY_MIN = 0.55  # Maximum penalty (most aggressive)
USER_TAG_SATURATION = 15     # Number of user tags before max penalty is applied


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
        {"flag": "--no-drop-overlap", "help": "Don't remove overlapping tags"},
        {"flag": "--no-infer-characters", "help": "Don't infer character names from AI"},
        {"flag": "--no-unload-models", "help": "Keep models loaded after batch (faster but uses more VRAM)", "default": "unloaded"},
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
    _torii_db: ClassVar[DanbooruCharacters | None] = None
    _models_loaded: ClassVar[bool] = False
    _model_instances: ClassVar[dict[str, Any]] = {}

    def __init__(
        self,
        threshold: float = 0.35,
        drop_overlap: bool = True,
        whitelist: list[str] | None = None,
        blacklist: list[str] | None = None,
        danbooru_only: bool = False,
        use_user_hints: bool = True,
        user_bonus: float = 1.0,
        ai_penalty: float = 0.66,
        infer_characters: bool = True,
        unload_models_after_batch: bool = True,
        user_tag_penalty_min: float = USER_TAG_PENALTY_MIN,
        user_tag_penalty_max: float = USER_TAG_PENALTY_MAX,
        user_tag_saturation: int = USER_TAG_SATURATION,
    ) -> None:
        self.threshold: float = threshold
        self.drop_overlap: bool = drop_overlap
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
        self._character_extractor: CharacterExtractor | None = None

    def name(self) -> str:
        return "tag:generate"

    def validate(self, context: ImageContext) -> bool:
        return context.image_path.exists()

    def process(self, context: ImageContext) -> ImageContext | None:
        """Generate tags for the image."""
        logger.debug(f"Processing: {context.image_path.name}")

        self._load_databases()
        self._load_models()

        image: Image.Image = context.load_image()

        user_tags: set[str] = self._extract_user_tags(context)
        user_rating: str | None = context.rating

        # Log user hints at DEBUG level
        if user_tags:
            logger.debug(f"User hints ({len(user_tags)}):")
            for i, tag in enumerate(sorted(user_tags)):
                logger.debug(f"  {i+1:3d}. {tag}")

        character_entries = context.character_entries
        user_characters = context.get_character_tags() if character_entries else []

        if character_entries:
            logger.debug(f"Character entries from context ({len(character_entries)}):")
            for entry in character_entries:
                logger.debug(f"  - {entry.tag} (source: {entry.source.name})")

        use_ai_characters = self.infer_characters

        if "original" in user_tags or "borrowed character" in user_tags:
            use_ai_characters = False
            logger.debug("Special character tags detected - disabling AI character detection")

        if character_entries:
            use_ai_characters = False
            logger.debug(f"User-provided characters exist ({len(character_entries)}) - AI character detection DISABLED")

        # === WARNING: Check for missing grounding tags ===
        # Initialize character extractor if not already done
        if self._character_extractor is None:
            self._character_extractor = CharacterExtractor()
        
        # Check if any character tags were provided but none found
        has_character_hint = any(
            tag.startswith("character:") or 
            tag in self._character_tags or 
            self._character_extractor.is_character_tag(tag)
            for tag in user_tags
        )
        
        if not character_entries and not has_character_hint:
            # Check if user provided any character hints at all
            if not any(tag.startswith("character:") for tag in user_tags):
                logger.warning(
                    f"No character tags found in user hints for {context.image_path.name}. "
                    "If this is intentional, add 'original' or 'borrowed_character' to suppress this warning."
                )
        elif not character_entries and has_character_hint:
            # User provided character hints but none were resolved
            logger.warning(
                f"Character hints provided but none resolved for {context.image_path.name}. "
                "Check tag spelling or database coverage."
            )

        ai_tags, ai_rating, ai_characters = self._run_inference(image)

        # Log AI inference results at DEBUG level
        logger.debug(f"AI inference results ({len(ai_tags)} total tags):")
        sorted_ai = sorted(ai_tags.items(), key=lambda x: -x[1])
        for i, (tag, conf) in enumerate(sorted_ai[:20]):
            logger.debug(f"  {i+1:3d}. {tag}: {conf:.3f}")
        if len(sorted_ai) > 20:
            logger.debug(f"  ... and {len(sorted_ai) - 20} more")

        if ai_characters:
            logger.debug(f"AI-inferenced characters ({len(ai_characters)}):")
            for char, conf in ai_characters.items():
                logger.debug(f"  - {char}: {conf:.3f}")

        result = context.copy()
        result.inferenced_tags = ai_tags

        main_tags = {}
        for tag, conf in ai_tags.items():
            if conf >= self.threshold:
                main_tags[tag] = conf

        combined_tags, character_tags = self._combine_tags(
            user_tags=user_tags,
            ai_tags=main_tags,
            ai_characters=ai_characters,
            user_characters=user_characters,
        )

        if use_ai_characters:
            added_ai_chars = []
            for char, conf in ai_characters.items():
                if conf >= 0.5:
                    normalized = self._normalize_character_tag(char)
                    if normalized and normalized not in character_tags:
                        character_tags.append(normalized)
                        added_ai_chars.append(f"{normalized} ({conf:.3f})")
            if added_ai_chars:
                logger.debug(f"Added AI-detected characters: {', '.join(added_ai_chars)}")

        final_tags = self._apply_filters(combined_tags)

        # === Show deltas: What changed ===
        original_tag_count = len(user_tags)
        final_tag_count = len(final_tags)
        
        # Tags that were added by AI (not in user hints)
        user_tag_set = {self._normalize_character_tag(t) for t in user_tags}
        added_by_ai = [t for t in final_tags if t not in user_tag_set]
        
        # Tags that were removed (in user hints but not in final)
        removed = [t for t in user_tags if t not in final_tags]
        
        # Tags that were kept
        kept = [t for t in final_tags if t in user_tag_set]
        
        # Characters added/kept
        final_char_tags = [e.tag for e in result.character_entries] if result.character_entries else []
        user_char_set = {self._normalize_character_tag(c) for c in user_characters}
        
        kept_chars = [c for c in final_char_tags if c in user_char_set] if character_entries else []
        added_chars = [c for c in final_char_tags if c not in user_char_set] if character_entries else []
        
        # Show summary at INFO level
        logger.info(
            f"Tag generation for {context.image_path.name}: "
            f"{original_tag_count} user tags → {final_tag_count} final tags"
        )
        
        if kept:
            logger.info(f"  Kept: {len(kept)} tags")
            logger.debug(f"    {', '.join(kept[:10])}{'...' if len(kept) > 10 else ''}")
        
        if added_by_ai:
            logger.info(f"  Added by AI: {len(added_by_ai)} tags")
            logger.debug(f"    {', '.join(added_by_ai[:10])}{'...' if len(added_by_ai) > 10 else ''}")
        
        if removed:
            logger.info(f"  Removed: {len(removed)} tags")
            logger.debug(f"    {', '.join(removed[:10])}{'...' if len(removed) > 10 else ''}")
        
        if kept_chars:
            logger.info(f"  Characters kept: {', '.join(kept_chars)}")
        
        if added_chars:
            logger.info(f"  Characters added: {', '.join(added_chars)}")
        
        # Show final tags at DEBUG level
        if final_tags:
            logger.debug(f"Final tags ({len(final_tags)}):")
            for i, tag in enumerate(final_tags):
                logger.debug(f"  {i+1:3d}. {tag}")

        torii_metadata = self._build_torii_metadata(
            tags=final_tags,
            character_tags=character_tags,
            general_tags=combined_tags,
            user_tags=user_tags,
        )

        result.set_tags(list(final_tags), section=1)

        if character_entries:
            result.character_entries = character_entries.copy()
        else:
            result.character_entries = []
            char_db = get_character_database()
            for char in character_tags:
                data = char_db.query(char)
                if data:
                    result.character_entries.append(
                        CharacterEntry.from_database(char, data)
                    )
                else:
                    result.character_entries.append(
                        CharacterEntry(
                            tag=char,
                            source=CharacterSource.EXTRACTED,
                        )
                    )

        if ai_rating or user_rating:
            result.rating = ai_rating or user_rating
            logger.info(f"  Rating: {result.rating}")

        result.metadata["char_p_tags"] = torii_metadata["char_p_tags"]
        result.metadata["char_descr"] = torii_metadata["char_descr"]
        result.metadata["torii_tags"] = torii_metadata["tags"]
        result.metadata["torii_characters"] = torii_metadata["characters"]

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
                    logger.error(f"Failed to load image {context.image_path.name}: {e}")

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
                logger.error(f"Failed to process {context.image_path.name}: {e}")
                results.append((idx, context))

        results.sort(key=lambda x: x[0])
        processed_contexts: list[ImageContext] = [r[1] for r in results]

        for pos, idx in enumerate(valid_indices):
            contexts[idx] = processed_contexts[pos]

        if self.unload_models_after_batch:
            self._unload_models()

        return contexts

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
                    logger.error(f"Failed to load image {context.image_path.name}: {e}")

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
                logger.error(f"Failed to process {context.image_path.name}: {e}")
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

        logger.info("Loading tag generation models...")

        try:
            # This import triggers model loading
            from imgutils.tagging import wd14, pixai
            from imgutils.generic.classify import ClassifyModel
            from imgutils.generic.yolo import YOLOModel

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
            logger.info("Tag generation models loaded")

        except Exception as e:
            logger.warning(f"Failed to load models: {e}")
            # Mark as loaded anyway to prevent repeated attempts
            cls._models_loaded = True

    @classmethod
    def _unload_models(cls) -> None:
        """Properly unload all imgutils models and clear caches."""
        if not cls._models_loaded:
            return

        logger.info("Unloading tag generation models...")

        try:
            # 1. Clear model instances from our internal cache
            for model_name, model_instance in cls._model_instances.items():
                if hasattr(model_instance, 'clear'):
                    try:
                        model_instance.clear()
                    except Exception as e:
                        logger.warning(f"Failed to clear {model_name}: {e}")

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
                    logger.debug(f"CUDA memory after cleanup: {allocated:.2f}MB allocated, {cached:.2f}MB cached")
                else:
                    logger.debug("CUDA memory fully released")

                # Force another GC pass after CUDA cleanup
                gc.collect()

            # 5. Clear our internal state
            cls._model_instances.clear()
            cls._models_loaded = False

            logger.info("Tag generation models unloaded")

        except Exception as e:
            logger.warning(f"Failed to unload models: {e}")
            cls._models_loaded = False

    @classmethod
    def _clear_imgutils_caches(cls) -> None:
        """Clear all imgutils cached functions."""
        import sys
        import inspect

        cleared = 0

        # Functions that we know are cached and used by our models
        known_cached_funcs = [
            # WD14
            ('imgutils.tagging.wd14', '_get_wd14_model'),
            ('imgutils.tagging.wd14', '_get_wd14_weights'),
            ('imgutils.tagging.wd14', '_get_wd14_labels'),
            # PixAI
            ('imgutils.tagging.pixai', '_open_onnx_model'),
            ('imgutils.tagging.pixai', '_open_tags'),
            ('imgutils.tagging.pixai', '_open_preprocess'),
            ('imgutils.tagging.pixai', '_open_default_category_thresholds'),
            # YOLO
            ('imgutils.generic.yolo', '_open_models_for_repo_id'),
            # Classify
            ('imgutils.generic.classify', '_open_models_for_repo_id'),
            # Booru YOLO
            ('imgutils.detect.booru_yolo', '_open_models_for_repo_id'),
            # Person detect
            ('imgutils.detect.person', '_open_models_for_repo_id'),
        ]

        for module_name, func_name in known_cached_funcs:
            try:
                # Import the module
                __import__(module_name)
                module = sys.modules[module_name]
                
                # Get the function
                if hasattr(module, func_name):
                    func = getattr(module, func_name)
                    if hasattr(func, 'cache_clear') and callable(func.cache_clear):
                        func.cache_clear()
                        cleared += 1
            except Exception:
                pass

        # Also try to find any other cached functions in imgutils modules
        for module_name, module in list(sys.modules.items()):
            if not module_name.startswith('imgutils'):
                continue

            for name, obj in inspect.getmembers(module):
                # Check if this is a cached function
                if hasattr(obj, 'cache_clear') and callable(obj.cache_clear):
                    try:
                        # Check if it was decorated with ts_lru_cache
                        # The decorator adds a __wrapped__ attribute
                        if hasattr(obj, '__wrapped__'):
                            obj.cache_clear()
                            cleared += 1
                    except Exception:
                        pass

        logger.debug(f"Cleared {cleared} imgutils caches")

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

        if cls._torii_db is None:
            cls._torii_db = DanbooruCharacters()

        logger.debug(
            f"Loaded {len(cls._general_tags)} general tags and "
            f"{len(cls._character_tags)} character tags"
        )

    # =========================================================================
    # Tag Extraction
    # =========================================================================

    def _extract_user_tags(self, context: ImageContext) -> set[str]:
        user_tags: set[str] = set()

        if not self.use_user_hints:
            return user_tags

        for tag in context.get_tags(section=1):
            if ',' in tag:
                parts = [p.strip() for p in tag.split(',') if p.strip()]
                for part in parts:
                    cleaned = self._clean_tag(part)
                    if cleaned:
                        user_tags.add(cleaned)
            else:
                cleaned = self._clean_tag(tag)
                if cleaned:
                    user_tags.add(cleaned)

        for tag in context.get_tags(section=0):
            if ',' in tag:
                parts = [p.strip() for p in tag.split(',') if p.strip()]
                for part in parts:
                    cleaned = self._clean_tag(part)
                    if cleaned:
                        user_tags.add(cleaned)
            else:
                cleaned = self._clean_tag(tag)
                if cleaned:
                    user_tags.add(cleaned)

        return user_tags

    def _get_character_extractor(self) -> CharacterExtractor:
        if self._character_extractor is None:
            self._character_extractor = CharacterExtractor()
        return self._character_extractor

    # =========================================================================
    # Character Tag Normalization
    # =========================================================================

    def _normalize_character_tag(self, tag: str) -> str:
        if not tag:
            return ""

        if tag.startswith("character:"):
            tag = tag[10:]

        tag = tag.lower()
        tag = tag.replace(" ", "_")

        return tag.strip("_ ")

    # =========================================================================
    # AI Inference
    # =========================================================================

    def _run_inference(
        self,
        image: Image.Image,
    ) -> tuple[dict[str, float], str | None, dict[str, float]]:
        from imgutils.tagging import wd14, pixai

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
            
            logger.debug(f"Extracted rating: {rating} (from {wd_ratings})")

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

        pixai_general = {
            self._normalize_tag(k): v for k, v in pixai_general.items()
        }
        pixai_characters = {
            self._normalize_tag(k): v for k, v in pixai_characters.items()
        }

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
        user_tags: set[str],
        ai_tags: dict[str, float],
        ai_characters: dict[str, float],
        user_characters: list[str],
    ) -> tuple[dict[str, float], list[str]]:
        ai_characters_from_tags: list[str] = []
        ai_general_tags: dict[str, float] = {}

        for tag, conf in ai_tags.items():
            normalized = self._normalize_character_tag(tag)
            if normalized and normalized in self._character_tags:
                if normalized not in ai_characters_from_tags:
                    ai_characters_from_tags.append(normalized)
            else:
                ai_general_tags[normalized] = conf

        normalized_user_tags = {self._normalize_character_tag(tag) for tag in user_tags}
        
        remaining_user_tags = normalized_user_tags.copy()
        for char in user_characters:
            if char in remaining_user_tags:
                remaining_user_tags.remove(char)

        general_tags: dict[str, float] = {}

        for tag, conf in ai_general_tags.items():
            general_tags[tag] = conf

        if remaining_user_tags:
            user_count = len(remaining_user_tags)
            
            penalty = (
                self.user_tag_penalty_max - 
                (self.user_tag_penalty_max - self.user_tag_penalty_min) * 
                min(user_count / self.user_tag_saturation, 1.0)
            )

            for tag in list(general_tags.keys()):
                if tag not in remaining_user_tags:
                    general_tags[tag] = general_tags[tag] * penalty

            for tag in remaining_user_tags:
                if tag in general_tags:
                    general_tags[tag] = min(1.0, general_tags[tag] / penalty)
                    general_tags[tag] = min(1.0, general_tags[tag] + self.user_bonus)
                else:
                    general_tags[tag] = 0.95

        character_tags: list[str] = []

        for char in user_characters:
            if char not in character_tags:
                character_tags.append(char)

        for char in ai_characters_from_tags:
            if char not in character_tags:
                character_tags.append(char)

        for char, conf in ai_characters.items():
            if conf >= 0.5:
                normalized = self._normalize_character_tag(char)
                if normalized and normalized not in character_tags:
                    character_tags.append(normalized)

        general_tags = {
            tag: conf
            for tag, conf in general_tags.items()
            if conf >= self.threshold
        }

        return general_tags, character_tags

    # =========================================================================
    # Filtering
    # =========================================================================

    def _apply_filters(self, tags: dict[str, float]) -> list[str]:
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

    # =========================================================================
    # Torii Metadata
    # =========================================================================

    def _build_torii_metadata(
        self,
        tags: list[str],
        character_tags: list[str],
        general_tags: dict[str, float],
        user_tags: set[str],
    ) -> dict[str, Any]:
        from imgutils.tagging import character

        torii_data: dict[str, Any] = {
            "tags": [],
            "characters": [],
            "char_p_tags": {"chars": {}, "skins": {}},
            "char_descr": {"chars": {}, "skins": {}},
        }

        underscore_tags: list[str] = [tag.replace(" ", "_") for tag in tags]

        if not character_tags:
            characteristics: list[str] = []
            for tag in underscore_tags:
                if character.is_basic_character_tag(tag):
                    characteristics.append(tag)

            for char_tag in characteristics:
                if char_tag in underscore_tags:
                    underscore_tags.remove(char_tag)

            torii_fix: list[str] = ["grabbing_another's_breast", "looking_over_eyewear"]
            fix_tags: list[str] = [t for t in characteristics if t in torii_fix]
            for fix_tag in fix_tags:
                characteristics.remove(fix_tag)
                if fix_tag not in underscore_tags:
                    underscore_tags.append(fix_tag)

            torii_data["char_p_tags"]["chars"]["DO NOT CAPTION CHARACTER NAME"] = (
                characteristics
            )

        else:
            for char_name in character_tags:
                payload: dict[str, str] | None = self._torii_db.query(char_name)
                if not payload:
                    continue

                popular_tags_str: str = payload.get("popular_tags", "[]")
                try:
                    popular_tags: list[str] = ast.literal_eval(popular_tags_str)
                except (ValueError, SyntaxError) as e:
                    logger.warning(f"Failed to parse popular_tags for {char_name}: {e}")
                    popular_tags = []

                popular_char_tags: list[str] = [
                    t for t in popular_tags
                    if character.is_basic_character_tag(t)
                ]

                for t in popular_char_tags:
                    underscore_form: str = t.replace(" ", "_")
                    if underscore_form in underscore_tags:
                        underscore_tags.remove(underscore_form)

                description: str = payload.get("description", "")

                torii_data["char_p_tags"]["chars"][char_name] = popular_char_tags
                torii_data["char_descr"]["chars"][char_name] = description

        underscore_tags = character.drop_basic_character_tags(underscore_tags)

        torii_data["tags"] = underscore_tags
        torii_data["characters"] = character_tags

        return torii_data

    def _clean_tag(self, tag: str) -> str:
        if not tag:
            return ""
        
        cleaned = tag.strip()
        cleaned = ' '.join(cleaned.split())
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = cleaned.strip('.,;:!?"\'')
        
        if not cleaned:
            return ""
        
        return cleaned

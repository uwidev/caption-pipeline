"""
TagResolveStep: Manage CLIP token limits.
"""

from math import ceil

from caption_pipeline.core.context import ImageContext
from caption_pipeline.core.help import step_help
from caption_pipeline.core.step import PipelineStep
from caption_pipeline.utils.logging_utils import log
from caption_pipeline.utils.tokenizer import get_tokenizer


@step_help(
    name="tag:resolve",
    description="Resolve tags to fit within CLIP token limits.",
    long_description="""CLIP models have a 77-token context window. This step ensures your tags
fit within this limit by intelligently adding or dropping tags based on the
configured strategy.

The 'drop' mode will only drop tags to fit within the limit.
The 'add' mode will add tags from the inferenced pool (tags collected during
TagGenerationStep that were below the main threshold).
The 'smart' mode will add or drop tags based on padding.

'--force-windows N' will force tags to fill exactly N CLIP windows by adding
or dropping tags as needed. Tags below the threshold may be added to prevent
excessive padding.

--keep-hints (default: True) ensures original hinted tags are never dropped.
Only AI-generated tags will be dropped when needed.""",
    options=[
        {"flag": "--mode {smart,drop,add}", "help": "Resolution strategy", "default": "drop"},
        {"flag": "--max-padding INT", "help": "Maximum allowed padding tokens", "default": "30"},
        {
            "flag": "--max-windows INT",
            "help": "Maximum number of CLIP windows (0 = no limit)",
            "default": "0",
        },
        {
            "flag": "--force-windows INT",
            "help": "Force exactly this many CLIP windows (adds or drops tags to hit this)",
            "default": "0",
        },
        {
            "flag": "--threshold FLOAT",
            "help": "Confidence threshold for extra tags (uses main threshold if not set)",
        },
        {
            "flag": "--max-tags INT",
            "help": "Maximum number of tags to keep (0 = no limit)",
            "default": "0",
        },
        {
            "flag": "--no-keep-hints",
            "help": "Don't preserve original hinted tags (allow them to be dropped)",
            "default": "keep hints",
        },
    ],
    example="tag:resolve --mode drop --max-padding 20",
)
class TagResolveStep(PipelineStep):
    """
    Resolve tags to fit within CLIP token limits.
    """

    def __init__(
        self,
        mode: str = "drop",
        max_padding: int = 30,
        max_windows: int = 0,
        force_windows: int = 0,
        window_size: int = 77,
        threshold: float | None = None,
        max_tags: int = 0,
        keep_hints: bool = True,
    ):
        self.mode = mode
        self.max_padding = max_padding
        self.max_windows = max_windows
        self.force_windows = force_windows
        self.window_size = window_size
        self.threshold = threshold
        self.max_tags = max_tags
        self.keep_hints = keep_hints

        self._tokenizer = None
        self._context: ImageContext | None = None

    def name(self) -> str:
        return "tag:resolve"

    def validate(self, context: ImageContext) -> bool:
        """Run if there are tags to resolve."""
        return bool(context.get_tags(section=1))

    def _get_original_tags(self, context: ImageContext) -> set[str]:
        """Get original hinted tags as a set (sections 0 and 1 only)."""
        original_flat = []
        for section in [0, 1]:
            if section < len(context.original_tags):
                original_flat.extend(context.original_tags[section])
        return set(original_flat)

    def _drop_tags_safely(self, tags: list[str], target_tokens: int, original_set: set[str]) -> list[str]:
        """
        Drop tags until token count is <= target_tokens.
        If keep_hints is True, skip dropping tags that are in original_set.
        """
        if not tags:
            return tags
        
        result = tags.copy()
        current_tokens = self._count_tokens(result)
        
        # Track which tags we've considered dropping
        # We'll iterate from the end (lowest confidence) and skip originals
        while current_tokens > target_tokens and len(result) > 1:
            # Find the last tag that is NOT in original_set (if keep_hints is True)
            dropped = False
            for i in range(len(result) - 1, -1, -1):
                tag = result[i]
                if self.keep_hints and tag in original_set:
                    # Skip original hint tags
                    continue
                # Drop this tag
                result.pop(i)
                dropped = True
                break
            
            if not dropped:
                # All remaining tags are original hints - can't drop any more
                log.debug(f"Cannot drop more tags: all remaining {len(result)} tags are original hints")
                break
            
            current_tokens = self._count_tokens(result)
        
        if self.keep_hints and len(result) < len(tags):
            dropped_count = len(tags) - len(result)
            log.debug(f"Dropped {dropped_count} AI-generated tags (kept original hints)")
        
        return result

    def process(self, context: ImageContext) -> ImageContext | None:
        """Resolve tags to fit within CLIP limits."""
        with log.section(f"Processing: {context.image_path.name}"):
            tags = context.get_tags(section=1)
            if not tags:
                return context

            # Store context for _resolve_add
            self._context = context

            # Lazy load tokenizer
            if self._tokenizer is None:
                self._tokenizer = get_tokenizer()

            # Get original tags for preservation
            original_set = self._get_original_tags(context) if self.keep_hints else set()
            
            if self.keep_hints and original_set:
                log.debug(f"Preserving {len(original_set)} original hinted tags")

            # Calculate current token usage
            current_tokens = self._count_tokens(tags)
            current_windows = self._get_window_count(tags)
            current_padding = self._get_padding(tags)

            log.debug(
                f"Token usage: {current_tokens} tokens, {current_windows} windows, "
                f"padding: {current_padding}, force_windows: {self.force_windows}"
            )

            original_count = len(tags)
            original_tokens = current_tokens

            # If force_windows is set, we need to add or drop to hit exactly that many windows
            if self.force_windows > 0:
                log.debug(f"Forcing exactly {self.force_windows} CLIP windows")
                resolved = self._resolve_force_windows(tags, original_set)
            else:
                # Normal resolution logic
                need_resolve = (
                    (self.max_windows and current_windows > self.max_windows)
                    or (current_padding > self.max_padding and current_tokens > self.window_size)
                )

                # For add mode, always try to add if we can
                if self.mode == "add" and not need_resolve:
                    if current_padding > 0 and current_tokens < self.window_size:
                        need_resolve = True
                        log.debug("Add mode: attempting to add tags despite no immediate need")

                if not need_resolve and self.mode != "add":
                    log.debug("No resolution needed")
                    return context

                # Resolve based on mode
                if self.mode == "drop":
                    resolved = self._resolve_drop(tags, original_set)
                elif self.mode == "add":
                    resolved = self._resolve_add(tags)
                else:  # smart
                    resolved = self._resolve_smart(tags, original_set)

            # Store resolved tags
            result = context.copy()
            result.set_tags(resolved, section=1)

            final_tokens = self._count_tokens(resolved)
            final_windows = self._get_window_count(resolved)
            final_padding = self._get_padding(resolved)
            final_count = len(resolved)

            # === Show deltas ===
            added = [t for t in resolved if t not in tags]
            removed = [t for t in tags if t not in resolved]
            
            log.info(
                f"{original_count} tags ({original_tokens} tokens) → {final_count} tags ({final_tokens} tokens)"
            )
            
            if self.keep_hints and removed:
                # Check if any removed tags were original hints
                removed_set = set(removed)
                original_removed = removed_set & original_set
                if original_removed:
                    log.warning(
                        f"Original hinted tags were dropped: {', '.join(sorted(original_removed))}"
                    )
            
            if removed:
                log.info(f"  Removed: {len(removed)} tags")
                log.debug(f"    {', '.join(removed[:10])}{'...' if len(removed) > 10 else ''}")
            
            if added:
                log.info(f"  Added: {len(added)} tags")
                log.debug(f"    {', '.join(added[:10])}{'...' if len(added) > 10 else ''}")
            
            if not removed and not added:
                log.debug("  No changes needed")
            
            log.debug(
                f"Resolved to {final_tokens} tokens, {final_windows} windows, "
                f"padding: {final_padding}"
            )
            
            return result

    def _resolve_force_windows(self, tags: list[str], original_set: set[str]) -> list[str]:
        """
        Force tags to fill exactly N CLIP windows.

        Uses ALL inferenced tags (regardless of threshold) to fill the windows.
        Tags are added in order of confidence.
        """
        target_windows = self.force_windows
        target_tokens_min = (target_windows - 1) * self.window_size + self.max_padding
        target_tokens_max = target_windows * self.window_size

        current_tokens = self._count_tokens(tags)

        log.debug(
            f"Force windows: target {target_windows} windows "
            f"({target_tokens_min}-{target_tokens_max} tokens), "
            f"current: {current_tokens} tokens"
        )

        # If we need to drop (too many tokens)
        if current_tokens > target_tokens_max:
            log.debug(f"Dropping tags from {current_tokens} to <= {target_tokens_max} tokens")
            return self._drop_tags_safely(tags, target_tokens_max, original_set)

        # If we need to add (too few tokens)
        if current_tokens < target_tokens_min:
            log.debug(f"Adding tags from {current_tokens} to {target_tokens_min} tokens")
            return self._resolve_add_to_target(tags, target_tokens_min)

        # Already in range
        log.debug("Already within target range")
        return tags

    def _resolve_drop_to_target(self, tags: list[str], target_tokens: int, original_set: set[str]) -> list[str]:
        """Drop tags until token count is <= target_tokens."""
        return self._drop_tags_safely(tags, target_tokens, original_set)

    def _resolve_add_to_target(self, tags: list[str], target_tokens: int) -> list[str]:
        """
        Add tags until token count reaches target_tokens.

        Uses ALL available inferenced tags (regardless of threshold) to fill the windows.
        Tags are added in order of confidence (highest first) to maximize quality.
        """
        # Get inferenced tags (from context or inference)
        inferenced_tags = self._get_inferenced_tags()
        if not inferenced_tags:
            log.debug("No inferenced tags available to add")
            return tags

        # Sort ALL tags by confidence descending (no threshold filter!)
        sorted_tags = sorted(inferenced_tags.items(), key=lambda x: -x[1])
        
        # Filter out tags already in the main list
        tags_set = set(tags)
        available = [(tag, conf) for tag, conf in sorted_tags if tag not in tags_set]
        
        if not available:
            log.debug("All inferenced tags already in main list")
            return tags

        log.debug(
            f"Adding tags from {len(available)} available tags "
            f"(highest conf: {available[0][1]:.3f}, lowest: {available[-1][1]:.3f})"
        )

        result = tags.copy()
        added = 0
        current_tokens = self._count_tokens(result)
        
        for tag, conf in available:
            result.append(tag)
            added += 1
            current_tokens = self._count_tokens(result)
            
            # Check if we've reached the target after each addition
            if current_tokens >= target_tokens:
                log.debug(
                    f"Reached target {target_tokens} tokens at {current_tokens} "
                    f"({added} tags added, lowest added conf: {conf:.3f})"
                )
                break

        if added > 0:
            log.debug(
                f"Added {added} tags from inferenced pool "
                f"({len(available)} available), final: {current_tokens} tokens"
            )

        return result

    def _count_tokens(self, tags: list[str]) -> int:
        """Count tokens for a tag list."""
        if not tags:
            return 0
        text = ", ".join(tags)
        tokens = self._tokenizer.encode(text, return_tensors="pt")
        return tokens.shape[1]

    def _get_window_count(self, tags: list[str]) -> int:
        """Get number of CLIP windows used."""
        if not tags:
            return 0
        return ceil(self._count_tokens(tags) / self.window_size)

    def _get_padding(self, tags: list[str]) -> int:
        """Get padding tokens in the current window."""
        if not tags:
            return self.window_size
        return self.window_size - (self._count_tokens(tags) % self.window_size)

    def _resolve_drop(self, tags: list[str], original_set: set[str]) -> list[str]:
        """Drop tags until within limits."""
        if not self._needs_drop(tags):
            return tags

        result = tags[:]
        while self._needs_drop(result) and len(result) > 1:
            # Find the last tag that is NOT in original_set (if keep_hints is True)
            dropped = False
            for i in range(len(result) - 1, -1, -1):
                tag = result[i]
                if self.keep_hints and tag in original_set:
                    # Skip original hint tags
                    continue
                # Drop this tag
                result.pop(i)
                dropped = True
                break
            
            if not dropped:
                # All remaining tags are original hints - can't drop any more
                break

        return result

    def _get_inferenced_tags(self) -> dict[str, float]:
        """
        Get inferenced tags from context or run inference.
        """
        if not self._context:
            return {}

        # Check if inferenced tags exist in context
        if self._context.inferenced_tags is not None:
            log.debug(f"Using {len(self._context.inferenced_tags)} inferenced tags from context")
            return self._context.inferenced_tags

        # If no inferenced tags, run inference
        log.debug("No inferenced tags found in context - running inference to gather them")
        return self._run_inference_for_tags()

    def _run_inference_for_tags(self) -> dict[str, float]:
        """
        Run inference on the image to gather all tags.

        Uses a very low threshold to capture all possible tags.
        """
        if not self._context:
            return {}

        try:
            from imgutils.tagging import wd14, pixai

            image = self._context.load_image()

            # Use a very low threshold to capture everything
            low_threshold = 0.01

            # Run WD14 inference
            _, _, wd_general = wd14.get_wd14_tags(
                image,
                "EVA02_Large",
                no_underline=True,
                general_threshold=low_threshold,
            )

            # Run PixAI inference
            pixai_general, _ = pixai.get_pixai_tags(
                image,
                "v0.9",
                thresholds={
                    "general": low_threshold,
                    "character": 0.1,
                },
            )

            # Convert underscores to spaces
            pixai_general = {k.replace("_", " "): v for k, v in pixai_general.items()}

            # Combine WD14 and PixAI general tags
            combined: dict[str, float] = {}
            for tag, conf in wd_general.items():
                combined[tag] = conf

            for tag, conf in pixai_general.items():
                combined[tag] = max(combined.get(tag, 0), conf)

            # Store in context for future use
            self._context.inferenced_tags = combined

            log.debug(f"Gathered {len(combined)} tags from inference (threshold: {low_threshold:.2f})")

            return combined

        except Exception as e:
            log.error(f"Failed to run inference for tags: {e}")
            return {}

    def _resolve_add(self, tags: list[str]) -> list[str]:
        """
        Add tags from the inferenced pool until hitting the limit.

        Uses ALL available tags (regardless of threshold) to fill the window.
        """
        if not self._context:
            return tags

        # Get inferenced tags (from context or inference)
        inferenced_tags = self._get_inferenced_tags()
        if not inferenced_tags:
            log.debug("No inferenced tags available to add")
            return tags

        # Sort ALL tags by confidence descending (no threshold filter!)
        sorted_tags = sorted(inferenced_tags.items(), key=lambda x: -x[1])
        
        # Filter out tags already in the main list
        tags_set = set(tags)
        available = [(tag, conf) for tag, conf in sorted_tags if tag not in tags_set]

        if not available:
            log.debug("All inferenced tags already in main list")
            return tags

        result = tags.copy()
        added = 0

        for tag, conf in available:
            result.append(tag)
            added += 1
            # Check if we've hit the limit after each addition
            if self._at_limit(result):
                break

        if added > 0:
            log.debug(
                f"Added {added} tags from inferenced pool "
                f"({len(available)} available)"
            )

        return result

    def _resolve_smart(self, tags: list[str], original_set: set[str]) -> list[str]:
        """Intelligently add or drop tags."""
        current_tokens = self._count_tokens(tags)

        # If we have room and are within the first window, try to add
        if current_tokens < self.window_size and current_tokens > self.max_padding:
            result = self._resolve_add(tags)
            if result != tags:
                return result

        # If we need to drop, drop
        if self._needs_drop(tags):
            return self._resolve_drop(tags, original_set)

        return tags

    def _needs_drop(self, tags: list[str]) -> bool:
        """Check if tags need to be dropped."""
        if not tags:
            return False

        windows = self._get_window_count(tags)
        padding = self._get_padding(tags)
        tokens = self._count_tokens(tags)

        if self.max_windows and windows > self.max_windows:
            return True

        if padding > self.max_padding and tokens > self.window_size:
            return True

        return False

    def _at_limit(self, tags: list[str]) -> bool:
        """Check if tags are at the limit."""
        windows = self._get_window_count(tags)
        padding = self._get_padding(tags)

        if self.max_windows and windows >= self.max_windows:
            return True

        if padding <= self.max_padding:
            return True

        return False

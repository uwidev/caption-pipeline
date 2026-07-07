# src/caption_pipeline/utils/tag_cache.py

import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests

from caption_pipeline.prompts.categorize_tag import CATEGORIZE_TAG_SYSTEM_PROMPT
from caption_pipeline.tools.tag_utils import sort_tags_by_object
from caption_pipeline.utils.logging_utils import log

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

CATEGORY_CODES = {
    "rating": "rate",
    "bodies": "count",
    "character name or series": "name",
    "body parts": "body",
    "wearables": "wear",
    "shot composition": "frame",
    "pose": "pose",
    "action": "act",
    "expressions": "exp",
    "effects": "boom",
    "atmosphere": "vibe",
    "environment": "env",
    "uncertain": "idk",
}
CODE_TO_CATEGORY = {v: k for k, v in CATEGORY_CODES.items()}
CATEGORY_NAMES = list(CATEGORY_CODES.keys())


class TagCategoryCache:
    _instance: Optional["TagCategoryCache"] = None

    def __new__(cls, cache_path: Path = Path("./tag_categories.txt")):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, cache_path: Path = Path("./tag_categories.txt")):
        if self._initialized:
            return
        self.cache_path: Path = cache_path
        self.cache: Dict[str, str] = {}  # tag (spaces, lowercase) -> full category name
        self._dirty: bool = False
        self._load()
        self._initialized = True

        # Defaults for fallback calls
        self._ollama_url: str = "http://localhost:11434/api/chat"
        self._model: str = "deepseek-r1:8b"
        self._temperature: float = 0.6
        self._timeout: int = 120

    def _load(self) -> None:
        """Load cache from file. File uses spaces, we store normalized (lowercase, spaces)."""
        if not self.cache_path.exists():
            log.debug(f"No cache file found at {self.cache_path}, starting fresh")
            return

        self.cache = {}
        current_category = None
        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("[") and line.endswith("]"):
                        cat_name = line[1:-1].strip()
                        current_category = self._normalize_category(cat_name)
                        continue
                    if current_category is not None:
                        # Normalize tag: lowercase, spaces
                        tag_normalized = self._normalize_tag(line)
                        self.cache[tag_normalized] = current_category
        except Exception as e:
            log.warning(f"Failed to load tag cache: {e}")
            self.cache = {}

    def save(self) -> None:
        """Write cache to file. Tags are stored with spaces (lowercase)."""
        if not self._dirty:
            return
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            grouped: dict[str, list[str]] = {cat: [] for cat in CATEGORY_NAMES}
            for tag, category in self.cache.items():
                grouped[category].append(tag)

            with open(self.cache_path, "w", encoding="utf-8") as f:
                for category in CATEGORY_NAMES:
                    tags = grouped.get(category, [])
                    if tags:
                        f.write(f"[{category}]\n")
                        # Use object-based sorting
                        for tag in sort_tags_by_object(tags):
                            f.write(f"{tag}\n")
                        f.write("\n")
            self._dirty = False
            log.debug(f"Saved {len(self.cache)} tag categories to {self.cache_path}")
        except Exception as e:
            log.error(f"Failed to save tag cache: {e}")

    def get(self, tag: str) -> Optional[str]:
        """Get category for a tag (normalized internally)."""
        norm = self._normalize_tag(tag)
        return self.cache.get(norm)

    def set(self, tag: str, category: str) -> None:
        """Store a tag->category mapping (normalized)."""
        norm_tag = self._normalize_tag(tag)
        canonical_cat = self._normalize_category(category)
        if self.cache.get(norm_tag) != canonical_cat:
            self.cache[norm_tag] = canonical_cat
            self._dirty = True

    def _normalize_tag(self, tag: str) -> str:
        """Normalize a tag: lowercase, replace underscores with spaces, strip."""
        return tag.lower().replace("_", " ").strip()

    def _normalize_category(self, category: str) -> str:
        cat_lower = category.lower().strip()
        if cat_lower in CATEGORY_CODES:
            return cat_lower
        if cat_lower in CODE_TO_CATEGORY:
            return CODE_TO_CATEGORY[cat_lower]
        for full_name in CATEGORY_NAMES:
            if cat_lower == full_name.lower():
                return full_name
        return "uncertain"

    def classify_tags_batch(
        self,
        tags: List[str],
        ollama_url: str = "http://localhost:11434/api/chat",
        model: str = "deepseek-r1:8b",
        temperature: float = 0.6,
        timeout: int = 120,
        batch_size: int = 20,
        max_retries: int = 3,
    ) -> Dict[str, str]:
        self._ollama_url = ollama_url
        self._model = model
        self._temperature = temperature
        self._timeout = timeout

        results = {}
        # Normalize all input tags for caching and matching
        tag_to_original = {self._normalize_tag(t): t for t in tags}
        normalized_tags = list(tag_to_original.keys())

        # Identify uncached tags (normalized form)
        uncached_norm = [nt for nt in normalized_tags if self.get(nt) is None]
        if not uncached_norm:
            self.save()
            return results

        # Track retry counts per normalized tag
        retry_counts: Dict[str, int] = {nt: 0 for nt in uncached_norm}
        pool = uncached_norm.copy()
        total_start = time.time()

        while pool:
            batch = pool[:batch_size]
            if not batch:
                break

            log.info(f"Processing batch: {len(batch)} tags (pool: {len(pool)})")
            batch_results = self._call_llm_batch(
                [tag_to_original[nt] for nt in batch],  # original form for prompt
                ollama_url,
                model,
                temperature,
                timeout,
            )

            # Convert results to normalized tags
            normalized_results = {}
            for original_tag, cat in batch_results.items():
                norm = self._normalize_tag(original_tag)
                normalized_results[norm] = cat
                self.set(norm, cat)

            # Update results with original tags
            for norm_tag, cat in normalized_results.items():
                original = tag_to_original.get(norm_tag, norm_tag)
                results[original] = cat

            # Find missing tags
            batch_set = set(batch)
            found_set = set(normalized_results.keys())
            missing = batch_set - found_set

            # Remove processed tags from pool
            pool = [t for t in pool if t not in batch_set]

            if missing:
                log.warning(f"Missing {len(missing)} tags from this batch: {missing}")
                for norm_tag in missing:
                    retry_counts[norm_tag] = retry_counts.get(norm_tag, 0) + 1
                    if retry_counts[norm_tag] > max_retries:
                        log.warning(
                            f"Max retries exceeded for tag '{norm_tag}', classifying individually"
                        )
                        original_tag = tag_to_original.get(norm_tag, norm_tag)
                        single_cat = self._call_llm_single(
                            original_tag, ollama_url, model, temperature, timeout
                        )
                        results[original_tag] = single_cat
                        self.set(norm_tag, single_cat)
                    else:
                        # Re-add to pool for retry
                        pool.append(norm_tag)

            self.save()
            log.debug(f"Batch completed, pool remaining: {len(pool)} tags")

        total_elapsed = time.time() - total_start
        if tags:
            log.info(
                f"All tags classified in {total_elapsed:.2f}s (avg {total_elapsed / len(tags):.2f}s/tag)"
            )
        self.save()
        return results

    def _call_llm_batch(
        self,
        tags: List[str],
        ollama_url: str,
        model: str,
        temperature: float,
        timeout: int,
    ) -> Dict[str, str]:
        tag_input = "\n".join(tags)
        messages = [
            {"role": "system", "content": CATEGORIZE_TAG_SYSTEM_PROMPT},
            {"role": "user", "content": tag_input},
        ]

        temp = temperature if temperature is not None else 0.6

        for attempt in range(2):
            payload = {
                "model": model,
                "messages": messages,
                "stream": False,
                "temperature": temp + (0.05 * attempt),
                "num_predict": 2048,
            }

            try:
                start_time = time.time()
                response = requests.post(ollama_url, json=payload, timeout=timeout)
                elapsed = time.time() - start_time
                log.debug(f"Batch request took {elapsed:.2f}s")

                if response.status_code != 200:
                    log.error(f"Ollama returned {response.status_code}: {response.text[:500]}")
                    response.raise_for_status()

                data = response.json()
                content = data.get("message", {}).get("content", "").strip()
                log.debug(f"Raw batch response ({len(content)} chars): {content[:200]}...")

                # Remove markdown code fences
                content = re.sub(r"^```\s*", "", content, flags=re.MULTILINE)
                content = re.sub(r"```$", "", content, flags=re.MULTILINE)

                result = self._parse_response(content, tags)
                log.debug(f"Parsed {len(result)} tags from batch response")
                return result

            except Exception as e:
                log.error(f"Batch call failed: {e}")
                if attempt == 0:
                    continue
                else:
                    return {}

        return {}

    def _parse_response(self, content: str, input_tags: List[str]) -> Dict[str, str]:
        """
        Robustly parse the LLM response into tag -> category mappings.
        Handles various formatting: with/without delimiters, extra text, backticks.
        """
        # Build a set of known category names (including codes) for detection
        category_names = set(CATEGORY_CODES.keys()) | set(CODE_TO_CATEGORY.values())
        cat_lower_map = {cat.lower(): cat for cat in category_names}
        for code, full in CODE_TO_CATEGORY.items():
            cat_lower_map[code] = full

        result = {}
        # Strip any enclosing markdown code fences
        content = re.sub(r"^```[\s\S]*?\n", "", content)
        content = re.sub(r"\n```$", "", content)

        lines = content.splitlines()
        current_category = None

        # We'll try to match each input tag at least once
        # Keep track of which tags have been assigned
        remaining_tags = set(input_tags)

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            # Remove bullet/number prefixes
            cleaned = re.sub(r"^[\s]*[-•*]\s+", "", stripped)
            cleaned = re.sub(r"^\d+\.\s+", "", cleaned)

            # Try to find a delimiter
            tag = None
            category = None
            for delim in [" -> ", " : ", " - ", ":", "-"]:
                if delim in cleaned:
                    parts = cleaned.split(delim, 1)
                    if len(parts) == 2:
                        candidate_tag = parts[0].strip()
                        candidate_cat = parts[1].strip()
                        # Check if candidate_cat is a known category
                        cat_lower = candidate_cat.lower()
                        if cat_lower in cat_lower_map:
                            category = cat_lower_map[cat_lower]
                            tag = candidate_tag
                            break
                        else:
                            # Maybe the category is the last word(s) after the delimiter
                            # For robustness, try to extract a category from candidate_cat
                            words = candidate_cat.split()
                            for w in words:
                                if w.lower() in cat_lower_map:
                                    category = cat_lower_map[w.lower()]
                                    tag = candidate_tag
                                    break
                            if category:
                                break
            # If no delimiter worked, try to guess category from line
            if not tag or not category:
                # Split line by spaces and see if any word is a category
                words = cleaned.split()
                # Look from the end: often the category is the last word(s)
                for i in range(len(words) - 1, -1, -1):
                    candidate = " ".join(words[i:])
                    if candidate.lower() in cat_lower_map:
                        category = cat_lower_map[candidate.lower()]
                        tag = " ".join(words[:i])
                        break
                if not tag and not category:
                    # Maybe the line is just a category header
                    if cleaned.lower() in cat_lower_map:
                        current_category = cat_lower_map[cleaned.lower()]
                        continue
                    if current_category:
                        # If we have a current category, assume the line is a tag
                        tag = cleaned
                        category = current_category

            # If we have a tag and category, store it
            if tag and category:
                # Normalize the tag (lowercase, underscores to spaces)
                norm_tag = self._normalize_tag(tag)
                # Remove any extra formatting (backticks, quotes)
                norm_tag = norm_tag.strip("`'\"")
                # If this tag is in the input list, assign it
                # But we'll accept any tag that matches one of the input tags (case-insensitive, spaces normalized)
                # Find the original input tag that matches this normalized form
                matched_original = None
                for inp in input_tags:
                    if self._normalize_tag(inp) == norm_tag:
                        matched_original = inp
                        break
                if matched_original:
                    result[matched_original] = category
                    remaining_tags.discard(matched_original)
                else:
                    # It might be a tag not in the input (extra output) – we can ignore or store anyway
                    # But we should only store if it's in the input list (to avoid polluting)
                    pass

        # After processing all lines, any remaining tags are missing
        if remaining_tags:
            log.warning(f"Missing tags after parsing: {remaining_tags}")
            # We'll leave them unassigned; caller will retry individually.

        return result

    def _call_llm_single(
        self,
        tag: str,
        ollama_url: str,
        model: str,
        temperature: float,
        timeout: int,
    ) -> str:
        result = self._call_llm_batch([tag], ollama_url, model, temperature, timeout)
        return result.get(tag, "uncertain")

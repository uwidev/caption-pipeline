"""
TagNaturalLanguageStep: Generate natural language captions using ToriiGate.

This step uses the ToriiGate-0.5 vision-language model to generate natural
language descriptions from images and their associated tags.
"""

import base64
import io
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Literal

import requests
from dotenv import load_dotenv
from PIL import Image

from caption_pipeline.core.context import ImageContext
from caption_pipeline.core.help import step_help
from caption_pipeline.core.step import PipelineStep
from caption_pipeline.prompts import TORIIGATE_PROMPTS
from caption_pipeline.utils.llama_server import LlamaServer, LlamaServerConfig
from caption_pipeline.utils.logging_utils import log, log_list_truncated, log_truncated, section
from caption_pipeline.utils.tag_db import get_display_name, query_character

load_dotenv()

# Server defaults
DEFAULT_SERVER_PORT = 8081
DEFAULT_SERVER_HOST = "127.0.0.1"
DEFAULT_SERVER_BINARY = "llama-server"
DEFAULT_SERVER_N_GPU_LAYERS = 999
DEFAULT_SERVER_FLASH_ATTN = True
DEFAULT_SERVER_CONTEXT_SIZE = 262144
DEFAULT_SERVER_IMAGE_MIN_TOKENS = 1024
DEFAULT_SERVER_CACHE_TYPE_K = "q8_0"
DEFAULT_SERVER_CACHE_TYPE_V = "q8_0"
DEFAULT_SERVER_LOG_VERBOSITY = 2
DEFAULT_SERVER_STARTUP_TIMEOUT = 60
DEFAULT_SERVER_SHUTDOWN_TIMEOUT = 10
DEFAULT_SERVER_CACHE_RAM = 0

# API defaults
DEFAULT_API_KEY = "not-needed"
DEFAULT_MODEL = "torii-gate-0.5"

# Caption defaults
DEFAULT_CAPTION_TYPE = "short"
DEFAULT_MAX_PIXELS = 1.0
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.5
DEFAULT_TIMEOUT = 120
DEFAULT_MAX_RETRIES = 3
DEFAULT_OUTPUT_SUFFIX = "-nl.txt"

# System prompt
SYSTEM_PROMPT = (
    "You are image captioning expert. Describe user's picture "
    "according to requested format and instructions."
)


@step_help(
    name="tag:nl",
    description="Generate natural language captions using ToriiGate-0.5.",
    long_description="""This step uses the ToriiGate-0.5 vision-language model to generate
descriptive captions from images.

The step manages the server lifecycle at the BATCH level - the server starts once
for all images in the batch and stops after processing completes.""",
    options=[
        {
            "flag": "--type {short,long,long_thoughts,long_thoughts_v2,json,json_comic,md_comic,min_structured_md,min_structured_json,chroma-style}",
            "help": "Caption format",
            "default": "short",
        },
        {"flag": "--model-path PATH", "help": "Path to ToriiGate .gguf model file"},
        {"flag": "--mmproj-path PATH", "help": "Path to ToriiGate mmproj file"},
        {"flag": "--port INT", "help": "Server port", "default": "8081"},
        {"flag": "--host HOST", "help": "Server host", "default": "127.0.0.1"},
        {"flag": "--log-file PATH", "help": "Log file for server output"},
        {
            "flag": "--no-auto-server",
            "help": "Don't manage server lifecycle (assume server is already running)",
            "default": "auto-manage",
        },
    ],
    example="tag:nl --type long --model-path ./torii-gate.gguf --mmproj-path ./mmproj.gguf",
)
class TagNaturalLanguageStep(PipelineStep):
    """
    Generate natural language captions using ToriiGate.

    Server management is at the BATCH level:
    - Server starts once before processing all images
    - Stays running for the entire batch
    - Stops after all images are processed
    """

    _STRUCTURED_TYPES: set[str] = {
        "long_thoughts",
        "long_thoughts_v2",
        "json",
        "json_comic",
        "md_comic",
        "min_structured_md",
        "min_structured_json",
        "chroma-style",
    }

    def __init__(
        self,
        # Server configuration
        model_path: Path | None = None,
        mmproj_path: Path | None = None,
        server_port: int = DEFAULT_SERVER_PORT,
        server_host: str = DEFAULT_SERVER_HOST,
        server_log_file: Path | None = None,
        auto_manage_server: bool = True,
        # API configuration
        api_url: str | None = None,
        api_key: str = DEFAULT_API_KEY,
        model: str = DEFAULT_MODEL,
        # Caption configuration
        caption_type: Literal[
            "short",
            "long",
            "long_thoughts",
            "long_thoughts_v2",
            "json",
            "json_comic",
            "md_comic",
            "min_structured_md",
            "min_structured_json",
            "chroma-style",
        ] = DEFAULT_CAPTION_TYPE,
        use_character_names: bool = True,
        include_tags: bool = True,
        include_character_list: bool = True,
        include_character_tags: bool = True,
        include_character_descriptions: bool = True,
        max_pixels: float = DEFAULT_MAX_PIXELS,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        timeout: int = DEFAULT_TIMEOUT,
        force: bool = False,
        output_suffix: str = DEFAULT_OUTPUT_SUFFIX,
        require_tags: bool = True,
        max_retries: int = DEFAULT_MAX_RETRIES,
        validate_response: bool = True,
        debug: bool = False,
    ) -> None:
        """
        Initialize the natural language generation step.

        All server configuration beyond the essentials uses defaults.
        """
        # Server configuration
        self.model_path = model_path
        if self.model_path is None:
            env_path = os.getenv("TORIIGATE_MODEL_PATH")
            if env_path:
                self.model_path = Path(env_path)
                log.debug(f"Loaded model_path from env: {self.model_path}")

        self.mmproj_path = mmproj_path
        if self.mmproj_path is None:
            env_path = os.getenv("TORIIGATE_MMPROJ_PATH")
            if env_path:
                self.mmproj_path = Path(env_path)
                log.debug(f"Loaded mmproj_path from env: {self.mmproj_path}")

        self.server_port = server_port
        self.server_host = server_host
        self.server_log_file = server_log_file
        self.auto_manage_server = auto_manage_server

        # Server defaults (not configurable via CLI)
        self.server_binary = DEFAULT_SERVER_BINARY
        self.server_n_gpu_layers = DEFAULT_SERVER_N_GPU_LAYERS
        self.server_flash_attn = DEFAULT_SERVER_FLASH_ATTN
        self.server_context_size = DEFAULT_SERVER_CONTEXT_SIZE
        self.server_image_min_tokens = DEFAULT_SERVER_IMAGE_MIN_TOKENS
        self.server_cache_type_k = DEFAULT_SERVER_CACHE_TYPE_K
        self.server_cache_type_v = DEFAULT_SERVER_CACHE_TYPE_V
        self.server_log_verbosity = DEFAULT_SERVER_LOG_VERBOSITY
        self.server_startup_timeout = DEFAULT_SERVER_STARTUP_TIMEOUT
        self.server_shutdown_timeout = DEFAULT_SERVER_SHUTDOWN_TIMEOUT
        self.server_cache_ram = DEFAULT_SERVER_CACHE_RAM

        # API configuration
        self.api_url = api_url or f"http://{server_host}:{server_port}/v1/chat/completions"
        self.api_key = api_key
        self.model = model

        # Caption configuration
        self.caption_type = caption_type
        self.use_character_names = use_character_names
        self.include_tags = include_tags
        self.include_character_list = include_character_list
        self.include_character_tags = include_character_tags
        self.include_character_descriptions = include_character_descriptions
        self.max_pixels = max_pixels
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.force = force
        self.output_suffix = output_suffix
        self.require_tags = require_tags
        self.max_retries = max_retries
        self.validate_response = validate_response

        # Runtime state
        self._server: LlamaServer | None = None
        self._system_prompt = SYSTEM_PROMPT
        self.debug = debug

    def name(self) -> str:
        """Return the step's unique identifier."""
        return "tag:natural_language"

    def validate(self, context: ImageContext) -> bool:
        """Validate that the context can be processed."""
        if not context.image_path.exists():
            return False
        if self.require_tags:
            return bool(context.get_tags(section=1))
        return True

    def _should_process(self, context: ImageContext) -> bool:
        """Check if this image needs NL captioning."""
        existing_nl = context.get_tags(section=2)
        if not existing_nl:
            return True
        if self.force:
            log.debug(f"Force enabled - regenerating NL caption for {context.image_path.name}")
            return True
        log.debug(f"NL caption already exists for {context.image_path.name} - skipping")
        return False

    def process_batch(self, contexts: list[ImageContext]) -> list[ImageContext]:
        """
        Process multiple contexts with server managed at batch level.

        The server starts ONCE for the entire batch and stays running until
        all images are processed. Metadata is prepared per-image, just before
        inference, so logs for each image are grouped together.
        """
        if not contexts:
            return contexts

        # Filter contexts that need processing
        valid_indices: list[int] = []
        skipped_indices: list[int] = []

        for idx, context in enumerate(contexts):
            if not self.validate(context):
                skipped_indices.append(idx)
                continue
            if not self._should_process(context):
                skipped_indices.append(idx)
                continue
            valid_indices.append(idx)

        if not valid_indices:
            log.info("No contexts need NL captioning (all already have captions)")
            return contexts

        log.info(f"Generating NL captions for {len(valid_indices)} images")

        results: list[tuple[int, ImageContext]] = []

        # Start the server ONCE for the ENTIRE batch
        if self.auto_manage_server:
            try:
                with self._create_server() as server:
                    self._server = server

                    # Process each image
                    for idx in valid_indices:
                        context = contexts[idx]
                        try:
                            result = self.process(context)
                            if result is not None:
                                results.append((idx, result))
                            else:
                                results.append((idx, context))
                        except Exception as e:
                            log.error(f"Failed to process {context.image_path.name}: {e}")
                            results.append((idx, context))

                    self._server = None
            except Exception as e:
                log.error(f"Server management failed: {e}")
                raise RuntimeError(f"Failed to start llama-server: {e}") from e
        else:
            # Server is managed externally
            for idx in valid_indices:
                context = contexts[idx]
                try:
                    result = self.process(context)
                    if result is not None:
                        results.append((idx, result))
                    else:
                        results.append((idx, context))
                except Exception as e:
                    log.error(f"Failed to process {context.image_path.name}: {e}")
                    results.append((idx, context))

        # Merge results back into original list
        results.sort(key=lambda x: x[0])
        processed_contexts = [r[1] for r in results]

        for pos, idx in enumerate(valid_indices):
            contexts[idx] = processed_contexts[pos]

        return contexts

    def process(self, context: ImageContext) -> ImageContext | None:
        """
        Core processing logic for a single image with retries.

        This is the single entry point for processing one image. It:
        1. Prepares metadata
        2. Handles retries for invalid responses
        3. Returns the processed context

        The server is assumed to be already running when this is called.
        """
        # Prepare metadata for THIS image (logs appear here)
        with section(f"Processing: {context.image_path.name}"):
            metadata = self._prepare_metadata(context)

            for attempt in range(self.max_retries):
                try:
                    if attempt > 0:
                        delay = 1.0 * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
                        log.debug(
                            f"Retry {attempt + 1}/{self.max_retries} after {delay:.1f}s delay"
                        )
                        time.sleep(delay)

                    # Encode image
                    image_data = self._encode_image(context)

                    # Prepare messages
                    messages = self._prepare_messages(metadata, image_data)

                    # Call API
                    caption = self._call_api(messages, timeout=self.timeout + (attempt * 15))

                    if caption is None:
                        if attempt < self.max_retries - 1:
                            log.warning(
                                f"Attempt {attempt + 1} failed: API returned None, retrying..."
                            )
                            continue
                        log.warning(
                            f"Failed to generate NL caption after {self.max_retries} attempts"
                        )
                        return context

                    # Validate response
                    if self.validate_response:
                        is_valid, error_msg = self._validate_response(
                            caption, context.image_path.name, attempt
                        )

                        if is_valid:
                            cleaned = self._clean_response(caption)
                            log_truncated("Generated NL caption", cleaned, max_len=128)

                            # Log prompt context at INFO level
                            tags = metadata.get("tags", [])
                            chars = metadata.get("characters", [])
                            char_str = f", {len(chars)} characters" if chars else ""
                            log.info(f"Context: {len(tags)} tags{char_str}")

                            result = context.copy()
                            result.set_tags([cleaned], section=2)
                            result.metadata["natural_language"] = cleaned
                            result.metadata["nl_attempts"] = attempt + 1
                            return result

                        if attempt < self.max_retries - 1:
                            log.warning(f"Attempt {attempt + 1} invalid: {error_msg}")
                            if self.debug:
                                log.debug(f"Invalid response preview: {caption[:200]}...")
                            continue

                        log.error(
                            f"All {self.max_retries} attempts failed for {context.image_path.name}"
                        )
                        if self.debug:
                            log.error(f"Final invalid response: {caption[:500]}...")
                        return context

                    # No validation - just use it
                    log.info(f"Generated NL caption for {context.image_path.name}:")
                    log.info(f"{caption[:200]}{'...' if len(caption) > 200 else ''}")
                    tags = metadata.get("tags", [])
                    chars = metadata.get("characters", [])
                    char_str = f", {len(chars)} characters" if chars else ""
                    log.info(f"Context: {len(tags)} tags{char_str}")

                    result = context.copy()
                    result.set_tags([caption], section=2)
                    result.metadata["natural_language"] = caption
                    return result

                except requests.exceptions.Timeout as e:
                    log.warning(f"Timeout on attempt {attempt + 1}/{self.max_retries}: {e}")
                    if attempt == self.max_retries - 1:
                        log.error(
                            f"All {self.max_retries} attempts timed out for {context.image_path.name}"
                        )
                        return context
                    continue
                except Exception as e:
                    log.error(f"Error on attempt {attempt + 1} for {context.image_path.name}: {e}")
                    if attempt == self.max_retries - 1:
                        return context
                    continue

            return context

    # =========================================================================
    # Server Management
    # =========================================================================

    def _create_server(self) -> LlamaServer:
        """Create a llama-server instance with configuration."""
        config = LlamaServerConfig(
            model_path=self.model_path,
            mmproj_path=self.mmproj_path,
            host=self.server_host,
            port=self.server_port,
            binary=self.server_binary,
            n_gpu_layers=self.server_n_gpu_layers,
            flash_attn=self.server_flash_attn,
            context_size=self.server_context_size,
            image_min_tokens=self.server_image_min_tokens,
            cache_type_k=self.server_cache_type_k,
            cache_type_v=self.server_cache_type_v,
            log_verbosity=self.server_log_verbosity,
            startup_timeout=self.server_startup_timeout,
            shutdown_timeout=self.server_shutdown_timeout,
            log_file=self.server_log_file,
            cache_ram=self.server_cache_ram,
        )
        return LlamaServer(config)

    # =========================================================================
    # API Calls
    # =========================================================================

    def _call_api(self, messages: list[dict[str, Any]], timeout: int | None = None) -> str | None:
        """
        Call the llama-server chat completions API with stateless behavior.

        Each request is independent and does not accumulate context.
        """
        timeout = timeout or self.timeout

        api_url = f"http://{self.server_host}:{self.server_port}/v1/chat/completions"

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": False,
            "cache_prompt": False,
            "num_predict": self.max_tokens,
            "slot_id": 0,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        log.debug(f"API request: {self.caption_type} caption, {self.max_tokens} max tokens")

        try:
            response = requests.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()

            result = response.json()
            content = result["choices"][0]["message"]["content"]
            log.debug(f"API response: {len(content)} chars")
            return content

        except requests.exceptions.ConnectionError as e:
            log.error(f"Connection error: {e}")
            log.error(f"Server URL: {api_url}")
            return None
        except requests.exceptions.Timeout as e:
            log.error(f"Request timeout: {e}")
            return None
        except requests.exceptions.RequestException as e:
            log.error(f"API request failed: {e}")
            if hasattr(e, "response") and e.response is not None:
                log.error(f"Response body: {e.response.text[:500]}")
            return None
        except (KeyError, IndexError) as e:
            log.error(f"Failed to parse API response: {e}")
            return None

    # =========================================================================
    # Image Encoding
    # =========================================================================

    def _encode_image(self, context: ImageContext) -> str:
        """Encode image to base64 with optional resizing."""
        img = context.load_image()

        current_pixels = img.width * img.height
        max_pixels_count = self.max_pixels * 1_000_000

        if current_pixels > max_pixels_count:
            scale = (max_pixels_count / current_pixels) ** 0.5
            new_width = int(img.width * scale)
            new_height = int(img.height * scale)

            try:
                resample = Image.Resampling.LANCZOS
            except AttributeError:
                resample = Image.LANCZOS  # type: ignore

            img = img.resize((new_width, new_height), resample)

        if img.mode != "RGB":
            img = img.convert("RGB")

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=95)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    # =========================================================================
    # Message Preparation
    # =========================================================================

    def _prepare_messages(
        self,
        metadata: dict[str, Any],
        image_data: str,
    ) -> list[dict[str, Any]]:
        """Prepare OpenAI-style messages."""
        user_query = self._make_user_query(metadata)

        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": self._system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                    },
                    {"type": "text", "text": user_query},
                ],
            },
        ]

    # =========================================================================
    # Metadata Preparation
    # =========================================================================

    def _prepare_metadata(self, context: ImageContext) -> dict[str, Any]:
        """Prepare metadata for the prompt."""
        tags = context.get_tags(section=1)

        if not tags and not self.require_tags:
            tags = context.get_tags(section=0)
            if not tags:
                log.debug("No tags found - generating NL caption from image only")
                tags = []

        # ===== DEBUG: Log grounding tags =====
        if self.debug and tags:
            log_list_truncated(tags, "Grounding tags", max_items=10, level="debug")

        character_tags = context.get_character_tags()

        char_p_tags = {"chars": {}, "skins": {}}
        char_descr = {"chars": {}, "skins": {}}

        # ===== DEBUG: Log character entries =====
        if self.debug:
            if character_tags:
                log_list_truncated(character_tags, "Characters", max_items=10, level="debug")
            else:
                log.debug("No characters found")

        # Handle unnamed/original character case
        if context.has_unnamed_character():
            tags_with_control = tags.copy() if tags else []
            if "original" not in tags_with_control:
                tags_with_control.append("original")
                if self.debug:
                    log.debug("Added 'original' control signal for unnamed character")

            metadata = {
                "tags": tags_with_control,
                "characters": [],
                "char_p_tags": {"chars": {}, "skins": {}},
                "char_descr": {"chars": {}, "skins": {}},
            }

            if self.debug:
                log.debug("Unnamed/original character detected - will describe without naming")

            return metadata

        # Handle named characters
        if character_tags:
            for character in character_tags:
                # Resolve skin to parent for display
                char_name = get_display_name(character)

                if self.debug:
                    log.debug(f"Loading data for character: '{char_name}'")

                # Query character data from booru_characters.csv
                data = query_character(char_name)

                if data:
                    popular_tags = data.get("popular_tags", [])
                    description = data.get("description", "")

                    if self.debug:
                        if popular_tags:
                            log_list_truncated(
                                popular_tags,
                                f"Popular tags for '{char_name}'",
                                max_items=10,
                                level="debug",
                            )
                        else:
                            log.debug(f"Popular tags for '{char_name}': (none)")

                    if self.debug and description:
                        log_truncated(
                            f"Description for '{char_name}'", description, max_len=64, level="debug"
                        )
                else:
                    # No data found - assume unnamed/original character
                    popular_tags = []
                    description = ""
                    if self.debug:
                        log.debug(f"No data found for '{char_name}' - treating as unnamed/original")

                char_p_tags["chars"][char_name] = popular_tags
                char_descr["chars"][char_name] = description

            tags_with_control = tags.copy() if tags else []

        else:
            # No characters at all - add control signal to prevent guessing
            tags_with_control = tags.copy() if tags else []
            if "original" not in tags_with_control:
                tags_with_control.append("original")
                if self.debug:
                    log.debug("Added 'original' control signal for ToriiGate")
            char_p_tags["chars"]["DO NOT CAPTION CHARACTER NAME"] = []
            char_descr["chars"]["DO NOT CAPTION CHARACTER NAME"] = ""
            if self.debug:
                log.debug("Added 'DO NOT CAPTION CHARACTER NAME' control signal")

        metadata = {
            "tags": tags_with_control,
            "characters": character_tags if character_tags else [],
            "char_p_tags": char_p_tags,
            "char_descr": char_descr,
        }

        # ===== INFO: Log context summary =====
        char_names = metadata["characters"]
        if char_names:
            char_str = f", characters: {', '.join(char_names)}"
            log.info(f"Context: {len(metadata['tags'])} tags{char_str}")
        else:
            log.info(f"Context: {len(metadata['tags'])} tags (no characters)")

        # ===== DEBUG: Log full metadata =====
        if self.debug:
            log.debug("Full metadata sent to model:")

            # Log tags with proper formatting
            tags_str = ", ".join(metadata["tags"])
            log_truncated("  tags", tags_str, max_len=64, level="debug")

            # Log characters
            if metadata["characters"]:
                chars_str = ", ".join(metadata["characters"])
                log_truncated("  characters", chars_str, max_len=64, level="debug")
            else:
                log.debug("  characters: (none)")

            # Log char_p_tags
            if metadata["char_p_tags"]["chars"]:
                log.debug("  char_p_tags:")
                for char_name, tags_list in metadata["char_p_tags"]["chars"].items():
                    if tags_list:
                        tags_str = ", ".join(tags_list)
                        log_truncated(f"    {char_name}", tags_str, max_len=64, level="debug")
                    else:
                        log.debug(f"    {char_name}: (none)")
            else:
                log.debug("  char_p_tags: (none)")

            # Log char_descr
            if metadata["char_descr"]["chars"]:
                log.debug("  char_descr:")
                for char_name, descr in metadata["char_descr"]["chars"].items():
                    if descr:
                        log_truncated(f"    {char_name}", descr, max_len=64, level="debug")
                    else:
                        log.debug(f"    {char_name}: (none)")
            else:
                log.debug("  char_descr: (none)")

        return metadata

    # =========================================================================
    # User Query Construction
    # =========================================================================

    def _make_user_query(self, metadata: dict[str, Any]) -> str:
        """Build the user query with all context."""
        tags = metadata.get("tags", []).copy()

        characters = metadata.get("characters", [])
        if not characters and "original" not in tags:
            tags.append("original")
            log.debug("Added 'original' control signal for ToriiGate")

        if tags:
            random.shuffle(tags)
            tags_string = ", ".join(tags)
        else:
            tags_string = "(No tags available - describe from image only)"

        user_request = "# Captioning format:\n"
        user_request += TORIIGATE_PROMPTS.get(self.caption_type, TORIIGATE_PROMPTS["short"])
        user_request += "\n"

        if self.include_tags:
            user_request += f"# Booru tags for the image\n[{tags_string}]\n\n"

        if self.use_character_names:
            if self.include_character_list:
                chars = metadata.get("characters", [])
                if chars:
                    chars_string = ", ".join(chars)
                    user_request += (
                        f"# Characters on picture:\nHere are names/tags for characters from the picture, "
                        f"make sure to use them: [{chars_string}].\n\n"
                    )

                char_p_tags = metadata.get("char_p_tags", {"chars": {}, "skins": {}})
                char_descr = metadata.get("char_descr", {"chars": {}, "skins": {}})

                if char_p_tags.get("chars") and self.include_character_tags:
                    user_request += "# Known traits for characters\n"
                    user_request += "Here are popular tags for each characters on picture:\n"
                    for c_name, c_tags in char_p_tags["chars"].items():
                        tags_s = ", ".join(c_tags)
                        user_request += f"{c_name}: [{tags_s}]\n"

                    if char_p_tags.get("skins"):
                        user_request += "Extra tags for characters skins:\n"
                        for c_name, c_tags in char_p_tags["skins"].items():
                            tags_s = ", ".join(c_tags)
                            user_request += f"{c_name}: [{tags_s}]\n"

                elif char_descr.get("chars") and self.include_character_descriptions:
                    user_request += (
                        "Here are general descriptions for each characters on the picture:\n"
                    )
                    for c_name, c_descr_text in char_descr["chars"].items():
                        user_request += f"## {c_name}\n{c_descr_text}\n\n"

                    if char_descr.get("skins"):
                        user_request += (
                            "Here are also descriptions for specific skin of characters:\n"
                        )
                        for c_name, c_descr_text in char_descr["skins"].items():
                            user_request += f"## {c_name}\n{c_descr_text}\n\n"
            else:
                user_request += "# Characters on picture:\nTry to recognize the characters in the picture and use their names.\n"
        else:
            user_request += "# Characters on picture:\nAvoid to guess names for characters.\n"

        return user_request

    # =========================================================================
    # Response Validation
    # =========================================================================

    def _validate_response(
        self,
        response: str,
        filename: str,
        attempt: int,
    ) -> tuple[bool, str | None]:
        """
        Validate a response for cleanliness.
        """
        if not response or len(response.strip()) < 5:
            return False, "Response is empty or too short"

        caption_type = self.caption_type

        # Check for control characters
        control_chars = re.findall(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", response)
        if control_chars:
            return False, f"Response contains control characters: {repr(control_chars[:5])}"

        # For structured types, allow newlines and markdown
        if caption_type in self._STRUCTURED_TYPES:
            if re.search(r"<[^>]+>", response) and not re.search(r"<format>|</format>", response):
                return False, "Response contains HTML tags outside format tags"
            if "<format>" in response and "</format>" not in response:
                return False, "Unclosed <format> tag"
            if "</format>" in response and "<format>" not in response:
                return False, "Unmatched </format> tag"
            if re.search(r"\n{4,}", response):
                return False, "Response contains excessive newlines"
            return True, None

        # For non-structured types - single paragraph
        cleaned = response.strip()
        if "\n\n" in cleaned:
            return False, "Response contains multiple paragraphs (newlines)"
        if re.search(r"^#+\s+", cleaned, re.MULTILINE):
            return False, "Response contains markdown headers"
        if "```" in cleaned:
            return False, "Response contains code blocks"
        if re.search(r"^[\s]*[-*•]\s+", cleaned, re.MULTILINE):
            return False, "Response contains bullet points"
        if re.search(r"^\s*\d+\.\s+", cleaned, re.MULTILINE):
            return False, "Response contains numbered list"
        if re.search(r"\{[^{}]*\}", cleaned):
            return False, "Response contains JSON-like structure"
        if re.search(r"\s{4,}", cleaned):
            return False, "Response contains excessive whitespace"
        if re.search(r"<[^>]+>", cleaned):
            return False, "Response contains HTML tags"
        if "<think>" in cleaned or "</think>" in cleaned:
            return False, "Response contains thinking tags"

        return True, None

    def _clean_response(self, response: str) -> str:
        """
        Clean up a response - always collapses to single paragraph.
        """
        if not response:
            return ""

        cleaned = response.strip()

        # Remove thinking tags
        cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL)

        # Remove format tags if present (keep content)
        cleaned = re.sub(r"<format>", "", cleaned)
        cleaned = re.sub(r"</format>", "", cleaned)

        # Always collapse to single paragraph
        cleaned = " ".join(cleaned.split())

        if cleaned.endswith(","):
            cleaned = cleaned[:-1]

        if cleaned and cleaned[-1] not in ".!?":
            cleaned = cleaned + "."

        return cleaned

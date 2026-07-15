"""
Tool: Query PixAI confidence for a specific tag across all images.
"""

import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm

from caption_pipeline.utils.logging_utils import configure_logging, log, section
from caption_pipeline.utils.image_utils import find_images_in_directory


def normalize_tag(tag: str) -> str:
    """Normalize a tag to match PixAI's output format (lowercase, spaces)."""
    return tag.lower().strip().replace("_", " ")


def get_tag_confidences(
    tag: str,
    input_dir: Path,
    output_path: Path,
    paths_path: Path,
    threshold: float = 0.01,
    recursive: bool = False,
    debug: bool = False,
) -> bool:
    """
    Query PixAI confidence for a specific tag across all images.

    The second output file contains the full paths to the corresponding
    caption files (*.txt) for each image, sorted by confidence descending.
    This file can be used with Neovim: `nvim -p $(cat paths_file)`.

    Returns:
        True if successful, False otherwise.
    """
    configure_logging(debug)

    with section(f"Tag confidence scan: '{tag}'"):
        # Find image files
        image_files = find_images_in_directory(input_dir, recursive=recursive)
        if not image_files:
            log.error("No image files found in the specified directory.")
            return False

        log.info(f"Found {len(image_files)} images to process.")

        # Normalize the target tag
        target_tag = normalize_tag(tag)
        log.info(f"Target tag normalized to: '{target_tag}'")

        # Load PixAI model
        try:
            from imgutils.tagging import pixai

            log.info("Loading PixAI model...")
            dummy = Image.new("RGB", (224, 224), color="white")
            pixai.get_pixai_tags(
                dummy,
                "v0.9",
                thresholds={"general": threshold, "character": threshold},
            )
            log.info("Model loaded successfully.")
        except Exception as e:
            log.error(f"Failed to load PixAI model: {e}")
            return False

        # Collect results
        results: list[tuple[Path, float]] = []
        log.info("Processing images...")
        for img_path in tqdm(image_files, desc="Inference", unit="img"):
            try:
                img = Image.open(img_path)
                pixai_general, pixai_characters = pixai.get_pixai_tags(
                    img,
                    "v0.9",
                    thresholds={"general": threshold, "character": threshold},
                )

                pixai_general = {normalize_tag(k): v for k, v in pixai_general.items()}
                pixai_characters = {normalize_tag(k): v for k, v in pixai_characters.items()}
                combined = {**pixai_general, **pixai_characters}

                conf = combined.get(target_tag, 0.0)
                results.append((img_path, conf))

            except Exception as e:
                log.warning(f"Failed to process {img_path.name}: {e}")
                results.append((img_path, 0.0))

        # Sort by confidence descending
        results.sort(key=lambda x: x[1], reverse=True)

        # Write outputs
        log.info(f"Writing results to {output_path} and {paths_path}")
        try:
            with open(output_path, "w", encoding="utf-8") as f_conf:
                f_conf.write("image,confidence\n")
                for img, conf in results:
                    f_conf.write(f"{img},{conf:.6f}\n")

            # Write the CAPTION file paths (not the image paths)
            with open(paths_path, "w", encoding="utf-8") as f_paths:
                for img, _ in results:
                    # Get the corresponding caption file
                    caption_path = img.with_suffix(".txt")
                    # Write absolute path, only if the file exists
                    if caption_path.exists():
                        f_paths.write(f"{caption_path.absolute()}\n")
                    else:
                        # If no caption file exists, write the image path (as a fallback)
                        log.debug(f"No caption file found for {img.name}, writing image path instead.")
                        f_paths.write(f"{img.absolute()}\n")

        except Exception as e:
            log.error(f"Failed to write output files: {e}")
            return False

        # Unload models
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            log.debug("Models unloaded.")
        except Exception:
            pass

        log.info(f"Done. Processed {len(results)} images.")
        return True

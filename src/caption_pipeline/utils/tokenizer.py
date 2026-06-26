"""
CLIP Tokenizer Wrapper.
"""

from pathlib import Path

from loguru import logger
from transformers import AutoTokenizer, PreTrainedTokenizerBase

_TOKENIZER: PreTrainedTokenizerBase | None = None


def get_tokenizer(model_path: Path = Path("./clip-vit-base-patch32/")) -> PreTrainedTokenizerBase:
    """
    Get the CLIP tokenizer (singleton).

    Args:
        model_path: Path to the CLIP model directory

    Returns:
        CLIP tokenizer instance
    """
    global _TOKENIZER

    if _TOKENIZER is None:
        try:
            _TOKENIZER = AutoTokenizer.from_pretrained(str(model_path))
            logger.info(f"Loaded CLIP tokenizer from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise

    return _TOKENIZER

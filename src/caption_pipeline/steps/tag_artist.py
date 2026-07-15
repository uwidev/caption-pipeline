"""
TagArtistStep: Infer artist using Kaloscope 2.0 (LSNet).
"""

import logging
from pathlib import Path
from typing import Any

import torch
import torchvision.transforms as transforms
from huggingface_hub import hf_hub_download
from PIL import Image

from caption_pipeline.core.context import ImageContext
from caption_pipeline.core.help import step_help
from caption_pipeline.core.step import PipelineStep
from caption_pipeline.utils.logging_utils import log, log_list_truncated, section
from caption_pipeline.utils.lsnet.lsnet_artist import LSNetArtist


@step_help(
    name="tag:artist",
    description="Infer artist using Kaloscope 2.0.",
    long_description="""This step uses Kaloscope 2.0 to identify the artist of an image.

If the user has already provided artist tags (via @prefix in section 0 or 1),
this step skips inference and uses the user-provided artists.

The model is loaded once per batch and unloaded after processing.

Kaloscope 2.0 is available at:
https://huggingface.co/heathcliff01/Kaloscope2.0""",
    options=[
        {
            "flag": "--threshold FLOAT",
            "help": "Minimum confidence to consider an artist (default: 0.1)",
            "default": "0.1",
        },
        {
            "flag": "--top-k INT",
            "help": "Number of top artists to add (0 = use threshold only, default: 3)",
            "default": "3",
        },
        {"flag": "--device {cpu,cuda}", "help": "Device to run inference on", "default": "auto"},
    ],
    example="tag:artist --threshold 0.2 --top-k 1 --device cuda",
)
class TagArtistStep(PipelineStep):
    # Class-level cache for the model
    _model: Any = None
    _model_loaded: bool = False
    _labels: list[str] | None = None
    _transform: Any = None
    _device: str = "cpu"

    def __init__(
        self,
        threshold: float = 0.1,
        top_k: int = 3,
        device: str = "auto",
        unload_after_batch: bool = True,
    ) -> None:
        self.threshold = threshold
        self.top_k = top_k
        self.device = device
        self.unload_after_batch = unload_after_batch

    def name(self) -> str:
        return "tag:artist"

    def validate(self, context: ImageContext) -> bool:
        return context.image_path.exists()

    def _should_process(self, context: ImageContext) -> bool:
        if context.artists:
            log.debug(f"Artist already provided: {', '.join(context.artists)} - skipping inference")
            return False
        return True

    def process_batch(self, contexts: list[ImageContext]) -> list[ImageContext]:
        if not contexts:
            return contexts

        valid_indices = []
        for idx, ctx in enumerate(contexts):
            if self.validate(ctx) and self._should_process(ctx):
                valid_indices.append(idx)

        if not valid_indices:
            log.info("No contexts need artist inference")
            return contexts

        log.info(f"Inferring artists for {len(valid_indices)} images")
        self._load_model()

        results = []
        for idx in valid_indices:
            ctx = contexts[idx]
            try:
                result = self.process(ctx)
                results.append((idx, result if result is not None else ctx))
            except Exception as e:
                log.error(f"Failed to process {ctx.image_path.name}: {e}")
                results.append((idx, ctx))

        results.sort(key=lambda x: x[0])
        for pos, (idx, ctx) in enumerate(results):
            contexts[idx] = ctx

        if self.unload_after_batch:
            self._unload_model()
        return contexts

    def process(self, context: ImageContext) -> ImageContext | None:
        with section(f"Processing: {context.image_path.name}"):
            if context.artists:
                return context

            image = context.load_image()
            artist_tags = self._run_inference(image)

            if not artist_tags:
                log.debug("No artist tags above threshold")
                return context

            # If top-k is set, take the top K artists by score
            if self.top_k > 0:
                sorted_items = sorted(artist_tags.items(), key=lambda x: x[1], reverse=True)
                artist_tags = dict(sorted_items[: self.top_k])

            # Logging
            is_debug = logging.getLogger().isEnabledFor(logging.DEBUG)
            if is_debug:
                formatted = [f"{tag}: {score:.3f}" for tag, score in artist_tags.items()]
                log_list_truncated(
                    items=formatted,
                    message=f"Artist tags (top {len(artist_tags)})",
                    max_items=-1,
                    level="debug",
                )
            else:
                top = list(artist_tags.items())[:10]
                log.debug(f"Raw artist tags ({len(artist_tags)} total):")
                for tag, score in top:
                    log.debug(f"  {tag}: {score:.3f}")

            result = context.copy()
            artist_names = list(artist_tags.keys())
            result.set_artists(artist_names)

            current = result.get_tags(1)
            for artist in artist_names:
                if artist not in current:
                    current.append(artist)
            result.set_tags(current, section=1)

            log_list_truncated(artist_names, f"Added artists ({len(artist_names)})")
            return result

    def _load_model(self) -> None:
        """Load Kaloscope 2.0 model, downloading checkpoint if missing."""
        if TagArtistStep._model_loaded:
            return

        log.info("Loading Kaloscope 2.0 model...")

        # Resolve device
        if self.device == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = self.device
        log.info(f"Using device: {self._device}")

        # Local checkpoint path (project‑relative)
        local_dir = Path("./models/Kalescope2.0")
        local_path = local_dir / "448-90.13" / "best_checkpoint.pth"

        # Download checkpoint if missing
        if not local_path.exists():
            log.info(f"Checkpoint not found at {local_path}, downloading (2.94 GB)...")
            local_dir.mkdir(parents=True, exist_ok=True)
            try:
                hf_hub_download(
                    repo_id="heathcliff01/Kaloscope2.0",
                    filename="448-90.13/best_checkpoint.pth",
                    local_dir=str(local_dir),
                    local_dir_use_symlinks=False,
                )
                log.info("Checkpoint downloaded.")
            except Exception as e:
                log.error(f"Download failed: {e}")
                raise

        # Load artist labels from CSV
        csv_path = Path("./artist_class_mapping.csv")
        labels = []
        if csv_path.exists():
            import csv

            try:
                with csv_path.open("r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        class_id = int(row.get("class_id", 0))
                        class_name = row.get("class_name", "").strip()
                        while len(labels) <= class_id:
                            labels.append(None)
                        labels[class_id] = class_name
                labels = [name for name in labels if name is not None]
                log.info(f"Loaded {len(labels)} artist names from {csv_path}")
            except Exception as e:
                log.warning(f"Failed to parse CSV: {e}. Using placeholders.")
                labels = [f"artist_{i}" for i in range(39261)]
        else:
            log.warning(f"CSV mapping not found at {csv_path}, using placeholder artist names.")
            labels = [f"artist_{i}" for i in range(39261)]

        TagArtistStep._labels = labels

        # Build model – parameters matching Kaloscope 2.0 XL/448
        model = LSNetArtist(
            img_size=448,
            patch_size=8,
            embed_dim=[192, 384, 576, 768],
            depth=[8, 12, 16, 20],
            num_heads=[6, 6, 6, 6],
            num_classes=39261,  # checkpoint has 39261 classes
            feature_dim=2048,
            use_projection=True,
        )

        # Load weights
        log.info(f"Loading weights from {local_path}...")
        state_dict = torch.load(local_path, map_location="cpu", weights_only=False)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif "model" in state_dict:
            state_dict = state_dict["model"]

        model.load_state_dict(state_dict, strict=True)
        model = model.to(self._device)
        model.eval()
        TagArtistStep._model = model

        # Preprocessing transform (448x448, ImageNet stats)
        TagArtistStep._transform = transforms.Compose(
            [
                transforms.Resize((448, 448), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        TagArtistStep._model_loaded = True
        log.info(f"Kaloscope 2.0 loaded on {self._device}")

    def _unload_model(self) -> None:
        if not TagArtistStep._model_loaded:
            return
        TagArtistStep._model = None
        TagArtistStep._model_loaded = False
        TagArtistStep._labels = None
        TagArtistStep._transform = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log.debug("Kaloscope 2.0 unloaded")

    def _run_inference(self, image: Image.Image) -> dict[str, float]:
        if not TagArtistStep._model_loaded:
            raise RuntimeError("Model not loaded")

        try:
            if image.mode != "RGB":
                image = image.convert("RGB")

            tensor = TagArtistStep._transform(image).unsqueeze(0).to(self._device)

            with torch.no_grad():
                output = TagArtistStep._model(tensor)
                probs = torch.softmax(output, dim=1)[0].cpu().numpy()

            results = {}
            labels = TagArtistStep._labels
            for i, score in enumerate(probs):
                if i >= len(labels) or score < self.threshold:
                    continue
                results[labels[i].replace("_", " ")] = float(score)

            return results

        except Exception as e:
            log.error(f"Inference failed: {e}")
            return {}

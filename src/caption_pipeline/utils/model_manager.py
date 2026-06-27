"""
ModelManager: Manages tag generation model lifecycle for batch processing.

Uses the ResourceManager pattern for consistent lifecycle management.
"""

import gc
import torch
from typing import Any

from PIL import Image

from caption_pipeline.core.resource_manager import ResourceManager


class ModelManager(ResourceManager):
    """
    Manages tag generation models (WD14, PixAI) with context manager support.
    
    Pattern:
        config = ModelConfig()
        with ModelManager(config):
            # Models are loaded
            for image in images:
                process_image(image)
        # Models are unloaded on exit
    
    This mirrors the ServerManager and OllamaManager patterns.
    """
    
    def __init__(self, config: Any = None) -> None:
        """
        Initialize the model manager.
        
        Args:
            config: Optional configuration (unused currently, kept for consistency)
        """
        super().__init__(config)
        self._model_instances: dict[str, Any] = {}
        self._models_loaded: bool = False
        
    def start(self) -> bool:
        """Load the tag generation models."""
        if self._ready:
            log.debug("Models already loaded")
            return True
        
        log.info("Loading tag generation models...")
        
        try:
            from imgutils.tagging import wd14, pixai
            from imgutils.generic.classify import ClassifyModel
            from imgutils.generic.yolo import YOLOModel
            
            # Force model loading by creating dummy prediction
            wd_model = ClassifyModel("wd14")
            self._model_instances["wd14"] = wd_model
            
            pixai_model = YOLOModel("pixai")
            self._model_instances["pixai"] = pixai_model
            
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
            
            self._ready = True
            self._started_by_us = True
            log.info("Tag generation models loaded")
            return True
            
        except Exception as e:
            log.warning(f"Failed to load models: {e}")
            return False
    
    def stop(self) -> bool:
        """Unload the tag generation models."""
        if not self._started_by_us:
            log.debug("Models not loaded by us - leaving as-is")
            return True
        
        if not self._ready:
            log.debug("Models already unloaded")
            self._started_by_us = False
            return True
        
        log.debug("Unloading tag generation models...")
        
        try:
            # 1. Clear model instances from our internal cache
            for model_name, model_instance in self._model_instances.items():
                if hasattr(model_instance, 'clear'):
                    try:
                        model_instance.clear()
                        log.debug(f"Cleared {model_name} model instance")
                    except Exception as e:
                        log.warning(f"Failed to clear {model_name}: {e}")
            
            # 2. Clear function-level caches (WD14 and PixAI)
            self._clear_function_caches()
            
            # 3. Force garbage collection
            gc.collect()
            log.debug("Garbage collection completed")
            
            # 4. Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                log.debug("CUDA synchronized")
                
                torch.cuda.empty_cache()
                log.debug("CUDA cache emptied")
                
                try:
                    torch.cuda.reset_peak_memory_stats()
                    log.debug("CUDA peak memory stats reset")
                except Exception:
                    pass
                
                allocated = torch.cuda.memory_allocated() / 1024 / 1024
                cached = torch.cuda.memory_reserved() / 1024 / 1024
                if allocated > 0 or cached > 0:
                    log.debug(f"CUDA memory after cleanup: {allocated:.2f}MB allocated, {cached:.2f}MB cached")
                else:
                    log.debug("CUDA memory fully released")
                
                gc.collect()
                log.debug("Post-CUDA garbage collection completed")
            
            # 5. Clear internal state
            self._model_instances.clear()
            self._ready = False
            self._started_by_us = False
            
            log.debug("Tag generation models unloaded")
            return True
            
        except Exception as e:
            log.warning(f"Failed to unload models: {e}")
            self._ready = False
            self._started_by_us = False
            return False
    
    def is_ready(self) -> bool:
        """Check if models are loaded."""
        return self._ready
    
    @property
    def is_loaded(self) -> bool:
        """Convenience property for model loaded state."""
        return self._ready
    
    def clear_inference_cache(self) -> None:
        """
        Clear per-image inference caches.
        
        This is called after each image while models stay loaded.
        """
        # Force garbage collection
        gc.collect()
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _clear_function_caches(self) -> None:
        """Clear function-level caches."""
        import sys
        
        cleared = 0
        errors = 0
        
        known_cached_funcs = [
            # WD14
            ('imgutils.tagging.wd14', '_get_wd14_model'),
            ('imgutils.tagging.wd14', '_get_wd14_weights'),
            ('imgutils.tagging.wd14', '_get_wd14_labels'),
            ('imgutils.tagging.wd14', '_open_denormalize_model'),
            # PixAI
            ('imgutils.tagging.pixai', '_open_onnx_model'),
            ('imgutils.tagging.pixai', '_open_tags'),
            ('imgutils.tagging.pixai', '_open_preprocess'),
            ('imgutils.tagging.pixai', '_open_default_category_thresholds'),
        ]
        
        for module_name, func_name in known_cached_funcs:
            try:
                __import__(module_name)
                module = sys.modules[module_name]
                
                if hasattr(module, func_name):
                    func = getattr(module, func_name)
                    if hasattr(func, 'cache_clear') and callable(func.cache_clear):
                        func.cache_clear()
                        cleared += 1
                        log.debug(f"Cleared cache for {module_name}.{func_name}")
            except Exception as e:
                errors += 1
                log.debug(f"Could not clear {module_name}.{func_name}: {e}")
        
        log.debug(f"Cleared {cleared} imgutils function-level caches ({errors} errors)")

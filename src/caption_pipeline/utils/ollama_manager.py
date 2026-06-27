"""
OllamaManager: Manages Ollama model lifecycle for batch processing.

Uses the ResourceManager pattern for consistent lifecycle management.
"""

from typing import Any

import requests

from caption_pipeline.core.resource_manager import ResourceManager
from caption_pipeline.utils.logging_utils import log


class OllamaConfig:
    """Configuration for Ollama resource."""

    def __init__(
        self,
        model: str,
        ollama_url: str = "http://localhost:11434/api/chat",
        keep_alive: int = 3600,
        load_timeout: int = 60,
        ping_timeout: int = 10,
    ) -> None:
        """
        Initialize Ollama configuration.

        Args:
            model: Ollama model name (e.g., "llama3.2:3b")
            ollama_url: Ollama API URL
            keep_alive: Seconds to keep model loaded after last request
            load_timeout: Timeout for model loading
            ping_timeout: Timeout for health checks
        """
        self.model: str = model
        self.ollama_url: str = ollama_url
        self.keep_alive: int = keep_alive
        self.load_timeout: int = load_timeout
        self.ping_timeout: int = ping_timeout


class OllamaManager(ResourceManager[OllamaConfig]):
    """
    Manages Ollama model lifecycle with context manager support.

    Pattern:
        config = OllamaConfig(model="llama3.2:3b")
        with OllamaManager(config):
            # Model is loaded and stays loaded
            for image in images:
                process_image(image)
        # Model is unloaded on exit

    This mirrors the ServerManager pattern used for llama-server.
    """

    def __init__(self, config: OllamaConfig) -> None:
        """
        Initialize the Ollama manager.

        Args:
            config: Ollama configuration
        """
        super().__init__(config)
        self._loaded: bool = False

    def start(self) -> bool:
        """
        Load the model into Ollama memory.

        Returns:
            True if loaded successfully, False otherwise
        """
        if self._ready:
            log.debug(f"Model '{self.config.model}' already loaded")
            return True

        log.info(f"Loading model '{self.config.model}' into Ollama...")

        # Send a minimal request to load the model
        try:
            response = requests.post(
                self.config.ollama_url,
                json={
                    "model": self.config.model,
                    "messages": [{"role": "user", "content": "ping"}],
                    "stream": False,
                    "keep_alive": self.config.keep_alive,
                    "options": {
                        "num_predict": 1,  # Minimal output
                    },
                },
                timeout=self.config.load_timeout,
            )

            if response.status_code == 200:
                self._ready = True
                self._started_by_us = True
                log.info(f"Model '{self.config.model}' loaded into Ollama")
                return True
            else:
                log.error(f"Failed to load model: HTTP {response.status_code}")
                log.error(f"Response: {response.text[:200]}")
                return False

        except requests.exceptions.ConnectionError:
            log.error(f"Could not connect to Ollama at {self.config.ollama_url}")
            log.error("Please ensure Ollama is running: 'ollama serve'")
            return False
        except requests.exceptions.Timeout:
            log.error(f"Timeout loading model '{self.config.model}'")
            return False
        except Exception as e:
            log.error(f"Failed to load model: {e}")
            return False

    def stop(self) -> bool:
        """
        Unload the model from Ollama memory.

        Returns:
            True if unloaded successfully, False otherwise
        """
        if not self._started_by_us:
            log.debug("Model not loaded by us - leaving as-is")
            return True

        if not self._ready:
            log.debug("Model already unloaded")
            self._started_by_us = False
            return True

        log.info(f"Unloading model '{self.config.model}' from Ollama...")

        # Send a request with keep_alive=0 to unload
        try:
            response = requests.post(
                self.config.ollama_url,
                json={
                    "model": self.config.model,
                    "messages": [{"role": "user", "content": "unload"}],
                    "stream": False,
                    "keep_alive": 0,  # ← Unload immediately
                },
                timeout=10,
            )

            self._ready = False
            self._started_by_us = False
            log.info(f"Model '{self.config.model}' unloaded from Ollama")
            return True

        except Exception as e:
            log.warning(f"Failed to unload model gracefully: {e}")
            self._ready = False
            self._started_by_us = False
            return False

    def is_ready(self) -> bool:
        """
        Check if the model is loaded and ready.

        Returns:
            True if loaded, False otherwise
        """
        return self._ready

    @property
    def is_loaded(self) -> bool:
        """Convenience property for model loaded state."""
        return self._ready

    def ping(self) -> bool:
        """
        Check if Ollama is responding.

        Returns:
            True if Ollama is responding, False otherwise
        """
        try:
            response = requests.get(
                "http://localhost:11434/api/tags",
                timeout=self.config.ping_timeout,
            )
            return response.status_code == 200
        except requests.RequestException:
            return False

    # Convenience method for the step
    def get_request_payload(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        """
        Get the base payload for Ollama requests.

        This ensures all requests use the same keep_alive value.

        Args:
            messages: The messages to send

        Returns:
            Dictionary payload for requests.post()
        """
        return {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
            "keep_alive": self.config.keep_alive,
        }

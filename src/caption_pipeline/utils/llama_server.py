"""
LlamaServer - A simple context manager for llama-server lifecycle management.
"""

# src/caption_pipeline/utils/llama_server.py

import os
import signal
import subprocess
import time
import socket
import threading
from pathlib import Path
from dataclasses import dataclass
from typing import Self

import requests
from loguru import logger


class LlamaServerError(Exception):
    """Raised when llama-server fails to start."""
    pass


@dataclass
class LlamaServerConfig:
    """Configuration for llama-server."""
    model_path: Path
    mmproj_path: Path
    host: str = "127.0.0.1"
    port: int = 8081
    binary: str = "llama-server"
    n_gpu_layers: int = 999
    flash_attn: bool = True
    context_size: int = 262144
    image_min_tokens: int = 1024
    cache_type_k: str = "q8_0"
    cache_type_v: str = "q8_0"
    log_verbosity: int = 0
    startup_timeout: int = 60
    shutdown_timeout: int = 10
    log_file: Path | None = None
    cache_ram: int = 0
    
    def build_command(self) -> list[str]:
        """Build the llama-server command line."""
        cmd = [
            self.binary,
            "-m", str(self.model_path),
            "--mmproj", str(self.mmproj_path),
            "--host", self.host,
            "--port", str(self.port),
            "-ngl", str(self.n_gpu_layers),
            "-c", str(self.context_size),
            "--image-min-tokens", str(self.image_min_tokens),
            "-ctk", self.cache_type_k,
            "-ctv", self.cache_type_v,
            "-lv", str(self.log_verbosity),
            "--cache-ram", str(self.cache_ram),
        ]
        
        if self.flash_attn:
            cmd.append("-fa")
            cmd.append("on")
        
        return cmd


class LlamaServer:
    """
    A simple context manager for llama-server.
    
    Uses a straightforward subprocess with proper cleanup.
    
    Usage:
        config = LlamaServerConfig(model_path=Path("model.gguf"), mmproj_path=Path("mmproj.gguf"))
        server = LlamaServer(config)
        with server:
            # Server is running
            response = requests.post("http://127.0.0.1:8081/v1/chat/completions", ...)
        # Server is shut down
    """
    
    def __init__(self, config: LlamaServerConfig):
        self.config = config
        self.process: subprocess.Popen | None = None
        self._log_file_handle = None
        self._is_ready = False
        self._pid: int | None = None
        self._stdout_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._startup_logs: list[str] = []  # Capture logs during startup
    
    def __enter__(self) -> Self:
        """Start the server when entering context."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop the server when exiting context."""
        self.stop()
    
    def start(self) -> bool:
        """
        Start the llama-server process.
        
        Raises:
            LlamaServerError: If the server fails to start or become ready
        """
        if self._is_ready:
            logger.debug("Server already running")
            return True
        
        if not self._validate_config():
            raise LlamaServerError("Server configuration is invalid - check model and mmproj paths")
        
        cmd = self.config.build_command()
        logger.info(f"Starting llama-server: {' '.join(cmd)}")
        
        # Prepare output destinations
        stdout_dest, stderr_dest = self._get_output_destinations()
        
        try:
            # Start the server as a subprocess
            self.process = subprocess.Popen(
                cmd,
                stdout=stdout_dest,
                stderr=stderr_dest,
                text=True,
                start_new_session=True,
                close_fds=True,
                bufsize=1,
            )
            
            self._pid = self.process.pid
            logger.debug(f"Server process started with PID: {self._pid}")
            
            # Start threads to log output if using pipes
            if stdout_dest == subprocess.PIPE:
                self._stdout_thread = threading.Thread(
                    target=self._log_output,
                    args=(self.process.stdout, "STDOUT"),
                    daemon=True
                )
                self._stdout_thread.start()
            
            if stderr_dest == subprocess.PIPE:
                self._stderr_thread = threading.Thread(
                    target=self._log_output,
                    args=(self.process.stderr, "STDERR"),
                    daemon=True
                )
                self._stderr_thread.start()
            
            # Wait for server to become ready
            if self._wait_for_ready():
                self._is_ready = True
                logger.info(f"Server is ready on port {self.config.port}")
                return True
            
            # Server failed to become ready - collect logs
            error_msg = self._get_failure_message()
            self._cleanup_process()
            raise LlamaServerError(error_msg)
            
        except FileNotFoundError as e:
            error_msg = (
                f"llama-server binary '{self.config.binary}' not found in PATH.\n"
                f"Please ensure llama-server is installed and available in your PATH.\n"
                f"Hint: Try 'which {self.config.binary}' to check if it's installed."
            )
            logger.error(error_msg)
            self._cleanup_process()
            raise LlamaServerError(error_msg) from e
            
        except PermissionError as e:
            error_msg = (
                f"Permission denied when trying to execute '{self.config.binary}'.\n"
                f"Please check that the binary has execute permissions."
            )
            logger.error(error_msg)
            self._cleanup_process()
            raise LlamaServerError(error_msg) from e
            
        except LlamaServerError:
            # Re-raise LlamaServerError as-is
            self._cleanup_process()
            raise
            
        except Exception as e:
            error_msg = f"Unexpected error while starting llama-server: {e}"
            logger.error(error_msg)
            import traceback
            logger.debug(traceback.format_exc())
            self._cleanup_process()
            raise LlamaServerError(error_msg) from e
    
    def stop(self) -> bool:
        """Stop the llama-server process."""
        if not self.process:
            logger.debug("No server process to stop")
            self._is_ready = False
            return True
        
        logger.info(f"Stopping llama-server (PID: {self._pid})...")
        
        try:
            # Check if process is still alive
            if self.process.poll() is not None:
                logger.debug(f"Process already terminated with exit code: {self.process.poll()}")
                self._cleanup_process()
                return True
            
            # Send SIGTERM to the process group
            if os.name != "nt":
                try:
                    os.killpg(os.getpgid(self._pid), signal.SIGTERM)
                    logger.debug(f"Sent SIGTERM to process group {os.getpgid(self._pid)}")
                except ProcessLookupError:
                    logger.debug("Process group already gone")
                    self._cleanup_process()
                    return True
                except OSError as e:
                    logger.debug(f"OS error sending SIGTERM: {e}")
                    # Fall back to terminating the process directly
                    self.process.terminate()
            else:
                self.process.terminate()
            
            # Wait for graceful shutdown
            try:
                stdout, stderr = self.process.communicate(
                    timeout=self.config.shutdown_timeout
                )
                if stdout:
                    logger.debug(f"Server stdout: {stdout[:500]}")
                if stderr:
                    logger.debug(f"Server stderr: {stderr[:500]}")
            except subprocess.TimeoutExpired:
                logger.warning("Server didn't shut down gracefully, forcing kill")
                if os.name != "nt":
                    try:
                        os.killpg(os.getpgid(self._pid), signal.SIGKILL)
                    except (ProcessLookupError, OSError):
                        self.process.kill()
                else:
                    self.process.kill()
                self.process.communicate()
            
            self._cleanup_process()
            logger.info("Server stopped")
            return True
            
        except ProcessLookupError:
            logger.debug("Process already terminated")
            self._cleanup_process()
            return True
        except OSError as e:
            # Bad file descriptor or other OS error - process is likely already gone
            if e.errno == 9:  # Bad file descriptor
                logger.debug("Process already terminated (bad file descriptor)")
                self._cleanup_process()
                return True
            logger.error(f"OS error stopping server: {e}")
            self._cleanup_process()
            return False
        except Exception as e:
            logger.error(f"Error stopping server: {e}")
            self._cleanup_process()
            return False

    def _cleanup_process(self) -> None:
        """Clean up the process and resources."""
        if self.process:
            try:
                # Check if process is still running before killing
                if self.process.poll() is None:
                    self.process.kill()
            except (ProcessLookupError, OSError) as e:
                # Process already gone, ignore
                if e.errno != 9:  # 9 = Bad file descriptor
                    logger.debug(f"Error killing process: {e}")
            finally:
                self.process = None
        
        self._pid = None
        
        # Close log file if open
        if self._log_file_handle:
            try:
                self._log_file_handle.close()
            except Exception:
                pass
            self._log_file_handle = None
        
        self._is_ready = False
        
    def is_ready(self) -> bool:
        """Check if the server is ready to accept requests."""
        if not self._is_ready:
            return False
        
        if not self._pid:
            return False
        
        # Quick socket check
        if not self._is_port_open():
            self._is_ready = False
            return False
        
        return True
    
    def _wait_for_ready(self, timeout: int | None = None) -> bool:
        """Wait for the server to become ready using health checks."""
        import requests
        
        timeout = timeout or self.config.startup_timeout
        start_time = time.time()
        
        # Try chat endpoint first (most reliable)
        endpoints = [
            "/v1/models",  # OpenAI-compatible endpoint
            "/health",     # llama-specific health endpoint
            "/",           # Root endpoint
        ]
        
        logger.debug(f"Waiting for server to become ready (timeout: {timeout}s)")
        
        while time.time() - start_time < timeout:
            # Check if process is still alive
            if self.process and self.process.poll() is not None:
                logger.error(f"Server process died (exit code: {self.process.poll()})")
                return False
            
            # Check if port is open
            if not self._is_port_open():
                time.sleep(0.5)
                continue
            
            # Port is open - try health endpoints
            for endpoint in endpoints:
                try:
                    url = f"{self.get_api_url()}{endpoint}"
                    response = requests.get(url, timeout=2.0)
                    if 200 <= response.status_code < 300:
                        logger.debug(f"Server ready (endpoint: {endpoint} returned {response.status_code})")
                        return True
                except requests.exceptions.ConnectionError:
                    # Server not ready yet
                    continue
                except requests.exceptions.Timeout:
                    # Server is busy loading the model
                    continue
                except Exception:
                    continue
            
            time.sleep(0.5)
        
        logger.error(f"Server failed to become ready within {timeout}s")
        return False
        
    def get_api_url(self) -> str:
        """Get the base API URL for the server."""
        return f"http://{self.config.host}:{self.config.port}"
    
    def get_chat_url(self) -> str:
        """Get the chat completions API URL."""
        return f"{self.get_api_url()}/v1/chat/completions"
    
    # =========================================================================
    # Private Methods
    # =========================================================================
    
    def _validate_config(self) -> bool:
        """Validate that the configuration is complete and files exist."""
        if not self.config.model_path:
            logger.error("Model path not set in configuration")
            return False
        
        if not self.config.model_path.exists():
            logger.error(f"Model file not found: {self.config.model_path}")
            return False
        
        if not self.config.mmproj_path:
            logger.error("MMProj path not set in configuration")
            return False
        
        if not self.config.mmproj_path.exists():
            logger.error(f"MMProj file not found: {self.config.mmproj_path}")
            return False
        
        return True
    
    def _get_output_destinations(self):
        """
        Get stdout/stderr destinations for the subprocess.
        
        If a log file is configured, write to it.
        Otherwise, pipe to Python for DEBUG logging.
        """
        if self.config.log_file:
            try:
                self.config.log_file.parent.mkdir(parents=True, exist_ok=True)
                self._log_file_handle = open(self.config.log_file, "w")
                return self._log_file_handle, self._log_file_handle
            except Exception as e:
                logger.warning(f"Could not open log file: {e}, falling back to DEBUG logging")
                return subprocess.PIPE, subprocess.PIPE
        else:
            return subprocess.PIPE, subprocess.PIPE
    
    def _log_output(self, pipe, name: str):
        """
        Read from a pipe and log each line at DEBUG level.
        
        Args:
            pipe: The pipe to read from
            name: "STDOUT" or "STDERR" for logging context
        """
        try:
            for line in iter(pipe.readline, ''):
                if line:
                    # Store startup logs for error reporting
                    if not self._is_ready:
                        self._startup_logs.append(line.rstrip())
                    logger.debug(f"[llama-server {name}] {line.rstrip()}")
        except Exception as e:
            logger.debug(f"Error reading {name}: {e}")
        finally:
            if pipe:
                try:
                    pipe.close()
                except Exception:
                    pass
    
    def _get_failure_message(self) -> str:
        """Generate a helpful error message from startup logs."""
        if not self._startup_logs:
            return (
                f"llama-server failed to become ready within {self.config.startup_timeout}s.\n"
                f"No output was captured from the server process.\n"
                f"Try running with --debug to see more details."
            )
        
        # Look for common error patterns
        error_lines = []
        warning_lines = []
        
        for line in self._startup_logs:
            if any(keyword in line.lower() for keyword in ['error', 'fail', 'exception', 'cuda', 'memory']):
                error_lines.append(line)
            elif any(keyword in line.lower() for keyword in ['warn', 'warning']):
                warning_lines.append(line)
        
        if error_lines:
            errors = '\n  '.join(error_lines[-5:])  # Last 5 error lines
            return (
                f"llama-server failed to start. Errors detected:\n"
                f"  {errors}\n"
                f"Full server output is available at DEBUG level."
            )
        
        # No obvious errors, show last few lines
        last_lines = '\n  '.join(self._startup_logs[-5:])
        return (
            f"llama-server failed to become ready within {self.config.startup_timeout}s.\n"
            f"Last lines from server:\n"
            f"  {last_lines}\n"
            f"Full server output is available at DEBUG level."
        )
    
    def _is_port_open(self) -> bool:
        """Check if the server port is open."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1.0)
            result = sock.connect_ex((self.config.host, self.config.port))
            sock.close()
            return result == 0
        except Exception:
            return False
    
    def __repr__(self) -> str:
        return (
            f"LlamaServer(host={self.config.host}, port={self.config.port}, "
            f"ready={self._is_ready}, pid={self._pid})"
        )

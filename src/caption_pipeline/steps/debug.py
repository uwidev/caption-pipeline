# src/caption_pipeline/steps/debug.py

import os
import socket

import psutil

from caption_pipeline.core.context import ImageContext
from caption_pipeline.core.step import PipelineStep
from caption_pipeline.utils.logging_utils import (
    log,
    section,
)


class DebugStep(PipelineStep):
    """Debug step to log system state."""

    def name(self) -> str:
        return "debug"

    def validate(self, context: ImageContext) -> bool:
        return True

    def process(self, context: ImageContext) -> ImageContext | None:
        with section(f"Processing: {context.image_path.name}"):
            log.debug("=" * 80)
            log.debug("DEBUG STATE")
            log.debug("=" * 80)

            # Process info
            log.debug(f"Current PID: {os.getpid()}")
            log.debug(f"Current PPID: {os.getppid()}")

            # Port check
            for port in [8081, 8082]:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(0.5)
                result = sock.connect_ex(("127.0.0.1", port))
                sock.close()
                log.debug(f"Port {port}: {'OPEN' if result == 0 else 'CLOSED'}")

            # Process list
            log.debug("Running processes:")
            for proc in psutil.process_iter(["pid", "name", "cmdline"]):
                try:
                    cmdline = " ".join(proc.info["cmdline"] or [])
                    if "llama" in cmdline.lower() or "server" in cmdline.lower():
                        log.debug(f"{proc.info['pid']}: {proc.info['name']} - {cmdline[:100]}")
                except:
                    pass

            log.debug("=" * 80)
            return context

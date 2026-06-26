# src/caption_pipeline/steps/debug.py

from caption_pipeline.core.context import ImageContext
from caption_pipeline.core.step import PipelineStep
from loguru import logger
import os
import psutil
import socket


class DebugStep(PipelineStep):
    """Debug step to log system state."""
    
    def name(self) -> str:
        return "debug"
    
    def validate(self, context: ImageContext) -> bool:
        return True
    
    def process(self, context: ImageContext) -> ImageContext | None:
        logger.debug(f"Processing: {context.image_path.name}")

        logger.debug("=" * 80)
        logger.debug("DEBUG STATE")
        logger.debug("=" * 80)
        
        # Process info
        logger.debug(f"Current PID: {os.getpid()}")
        logger.debug(f"Current PPID: {os.getppid()}")
        
        # Port check
        for port in [8081, 8082]:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.5)
            result = sock.connect_ex(('127.0.0.1', port))
            sock.close()
            logger.debug(f"Port {port}: {'OPEN' if result == 0 else 'CLOSED'}")
        
        # Process list
        logger.debug("Running processes:")
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = ' '.join(proc.info['cmdline'] or [])
                if 'llama' in cmdline.lower() or 'server' in cmdline.lower():
                    logger.debug(f"  {proc.info['pid']}: {proc.info['name']} - {cmdline[:100]}")
            except:
                pass
        
        logger.debug("=" * 80)
        return context

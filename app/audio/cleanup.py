import os
import shutil
import logging

logger = logging.getLogger("audio_pipeline.cleanup")

def cleanup_path(path: str):
    """
    Safely deletes a file or directory from the server.
    Logs warning messages instead of raising exceptions if something fails,
    ensuring that errors in cleanup do not interrupt API responses.
    """
    if not path or not os.path.exists(path):
        logger.debug(f"Cleanup skipped: path does not exist '{path}'")
        return
    
    try:
        if os.path.isdir(path):
            shutil.rmtree(path)
            logger.info(f"Successfully cleaned up directory tree: {path}")
        else:
            os.remove(path)
            logger.info(f"Successfully cleaned up file: {path}")
    except Exception as e:
        logger.warning(f"Resilient cleanup encountered an issue for '{path}': {e}")

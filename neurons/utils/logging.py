"""
Logging utilities.
"""

import logging
import sys
from typing import Optional


def setup_logging(
    level: int = logging.INFO,
    format_string: Optional[str] = None,
    log_file: Optional[str] = None
) -> None:
    """
    Setup logging configuration for the library.
    
    Args:
        level: Logging level. Default: logging.INFO
        format_string: Custom format string. If None, uses default format.
        log_file: Path to log file. If None, logs only to console.
        
    Examples:
        >>> setup_logging(level=logging.DEBUG)
        >>> setup_logging(level=logging.INFO, log_file='training.log')
    """
    if format_string is None:
        format_string = (
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file is not None:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=handlers,
        force=True
    )
    
    # Set library logger level
    logger = logging.getLogger('neurons')
    logger.setLevel(level)

import logging
import sys
import os

def setup_logger(name: str, log_dir: str, level=logging.INFO) -> logging.Logger:
    """Configures a robust logger that outputs to console and a file."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if logger.hasHandlers():
        logger.handlers.clear()
        
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File Handler
    log_file = os.path.join(log_dir, 'training.log')
    try:
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    except Exception as e:
        logger.error(f"Failed to set up file handler for logger: {e}")
        
    return logger
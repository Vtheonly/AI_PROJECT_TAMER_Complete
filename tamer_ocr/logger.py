import logging
import sys
import os

def setup_logger(name: str, log_dir: str = None, level=logging.INFO) -> logging.Logger:
    """
    Configures a robust logger that outputs to console and a file.
    
    Fixes:
    - Prevents duplicate log entries in interactive environments (Colab/Kaggle).
    - Ensures all sub-modules inherit the same configuration.
    - Handles file system permission errors gracefully.
    """
    # Get the base name (e.g., "TAMER") to configure the parent logger
    base_name = name.split('.')[0] if '.' in name else name
    root_logger = logging.getLogger(base_name)
    root_logger.setLevel(level)

    # FIX: Prevent adding handlers multiple times if the cell is re-run
    if not root_logger.hasHandlers():
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # 1. Console Handler (Prints to Jupyter Notebook / Colab / Terminal)
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(formatter)
        root_logger.addHandler(ch)

        # 2. File Handler (Saves to logs/training.log for persistence)
        if log_dir:
            try:
                os.makedirs(log_dir, exist_ok=True)
                log_file = os.path.join(log_dir, 'training.log')
                fh = logging.FileHandler(log_file, encoding='utf-8')
                fh.setLevel(level)
                fh.setFormatter(formatter)
                root_logger.addHandler(fh)
            except Exception as e:
                # Fallback: if file logging fails, print a warning but don't crash
                print(f"Warning: Failed to set up file handler at {log_dir}: {e}")

    # Return the specific child logger requested (e.g., "TAMER.Trainer")
    return logging.getLogger(name)
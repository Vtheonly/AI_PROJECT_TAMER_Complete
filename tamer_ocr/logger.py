import logging
import sys
import os

def setup_logger(name: str, log_dir: str, level=logging.INFO) -> logging.Logger:
    """Configures a robust logger that outputs to console and a file."""
    # CRITICAL FIX: Configure the base "TAMER" logger so all sub-modules 
    # (Trainer, Preprocessor, Downloader) inherit the console output.
    base_name = name.split('.')[0] if '.' in name else name
    root_logger = logging.getLogger(base_name)
    root_logger.setLevel(level)

    # Prevent adding handlers multiple times if called again
    if not root_logger.hasHandlers():
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Console Handler (Prints to Jupyter Notebook / Colab)
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(formatter)
        root_logger.addHandler(ch)

        # File Handler (Saves to logs/training.log)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, 'training.log')
            try:
                fh = logging.FileHandler(log_file)
                fh.setLevel(level)
                fh.setFormatter(formatter)
                root_logger.addHandler(fh)
            except Exception as e:
                print(f"Failed to set up file handler for logger: {e}")

    # Return the specific child logger requested (e.g., "TAMER.Trainer")
    return logging.getLogger(name)
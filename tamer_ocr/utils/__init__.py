"""
TAMER OCR Utils Module.

Metrics are always importable (no torch dependency).
Checkpoint utils are lazy (require torch).
"""

from .metrics import calculate_metrics, compute_batch_metrics, edit_distance

def __getattr__(name):
    _LAZY_IMPORTS = {
        'save_checkpoint': '.checkpoint',
        'load_checkpoint': '.checkpoint',
        'backup_to_drive': '.checkpoint',
        'push_to_huggingface': '.checkpoint',
    }
    if name in _LAZY_IMPORTS:
        import importlib
        module = importlib.import_module(_LAZY_IMPORTS[name], package='tamer_ocr.utils')
        return getattr(module, name)
    raise AttributeError(f"module 'tamer_ocr.utils' has no attribute {name}")

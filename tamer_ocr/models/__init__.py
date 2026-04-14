"""
TAMER OCR Models Module.

All imports are lazy to avoid requiring torch at package import time.
"""

def __getattr__(name):
    _LAZY_IMPORTS = {
        'TAMERModel': '.tamer',
        'SwinEncoder': '.encoder',
        'TransformerDecoder': '.decoder',
        'PositionalEncoding1D': '.attention',
        'PositionalEncoding2D': '.attention',
    }
    if name in _LAZY_IMPORTS:
        import importlib
        module = importlib.import_module(_LAZY_IMPORTS[name], package='tamer_ocr.models')
        return getattr(module, name)
    raise AttributeError(f"module 'tamer_ocr.models' has no attribute {name}")

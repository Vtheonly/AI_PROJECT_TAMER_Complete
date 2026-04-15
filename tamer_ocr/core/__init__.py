"""
TAMER OCR Core Module.

All imports are lazy to avoid requiring torch at package import time.
"""

def __getattr__(name):
    _LAZY_IMPORTS = {
        'Trainer': '.trainer',
        'LabelSmoothedCELoss': '.losses',
        'StructureAwareLoss': '.losses',
        'beam_search': '.inference',
        'greedy_decode': '.inference',
    }
    if name in _LAZY_IMPORTS:
        import importlib
        module = importlib.import_module(_LAZY_IMPORTS[name], package='tamer_ocr.core')
        return getattr(module, name)
    raise AttributeError(f"module 'tamer_ocr.core' has no attribute {name}")

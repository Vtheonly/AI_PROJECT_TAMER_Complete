"""
TAMER OCR Data Module.

All imports are lazy to avoid requiring torch at package import time.
Use direct imports for specific modules:
    from tamer_ocr.data.tokenizer import LaTeXTokenizer
    from tamer_ocr.data.dataset import MathDataset, get_collate_fn
"""

def __getattr__(name):
    _LAZY_IMPORTS = {
        'MathDataset': '.dataset',
        'get_collate_fn': '.dataset',
        'LaTeXTokenizer': '.tokenizer',
        'normalize_latex': '.latex_normalizer',
        'normalize_corpus': '.latex_normalizer',
        'should_discard': '.latex_normalizer',
        'TemperatureSampler': '.sampler',
        'MultiDatasetBatchSampler': '.sampler',
        'get_temperature_for_step': '.sampler',
        'DatasetPreprocessor': '.preprocessor',
        'create_data_manager': '.data_manager',
        'DatasetParser': '.parser',
        'create_parser': '.parser',
        'get_train_augmentation': '.augmentation',
        'get_val_augmentation': '.augmentation',
        'validate_before_training': '.validator',
        'validate_samples': '.validator',
    }
    if name in _LAZY_IMPORTS:
        import importlib
        module = importlib.import_module(_LAZY_IMPORTS[name], package='tamer_ocr.data')
        return getattr(module, name)
    raise AttributeError(f"module 'tamer_ocr.data' has no attribute {name}")

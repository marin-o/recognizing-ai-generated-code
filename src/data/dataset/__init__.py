# Dataset modules with lazy loading to avoid loading models at import time

def __getattr__(name):
    """Lazy loading of dataset classes to avoid loading models unnecessarily."""
    if name == 'AIGCodeSet':
        from .aigcodeset import AIGCodeSet
        return AIGCodeSet
    elif name == 'AIGCodeSet_WithCSTFeatures':
        from .aigcodeset_cst import AIGCodeSet_WithCSTFeatures
        return AIGCodeSet_WithCSTFeatures
    elif name == 'AIGCodeSet_Levenshtein':
        from .aigcodeset_levenshtein import AIGCodeSet_Levenshtein
        return AIGCodeSet_Levenshtein
    elif name == 'CoDeTM4':
        from .codet_m4 import CoDeTM4
        return CoDeTM4
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    'AIGCodeSet',
    'AIGCodeSet_WithCSTFeatures', 
    'AIGCodeSet_Levenshtein',
    'CoDeTM4'
]

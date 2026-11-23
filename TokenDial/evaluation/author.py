"""Load unmodified supplied source; reuse its low-level frame/model processing."""
import importlib.util
from pathlib import Path
import sys


def load_author(filename='evaluate_sliders.py'):
    path = Path(__file__).parent / 'vendor' / filename
    name = '_tokendial_author_' + path.stem
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]

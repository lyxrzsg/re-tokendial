from importlib import import_module
from .internvideo2 import pretrain_internvideo2_1b_patch14_224, pretrain_internvideo2_6b_patch14_224
def __getattr__(name):
    modules = {"internvl_clip_6b": "internvl_clip_vision", "InternVideo2": "internvideo2_clip_vision",
               "LLaMA": "internvideo2_clip_text", "Tokenizer": "internvideo2_clip_text",
               "TextTransformer": "mobileclip", "ClipTokenizer": "mobileclip"}
    if name not in modules:
        raise AttributeError(name)
    return getattr(import_module("." + modules[name], __name__), name)

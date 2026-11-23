from importlib import import_module
__all__ = ["InternVideo2_CLIP", "InternVideo2_Stage2"]
def __getattr__(name):
    modules = {"InternVideo2_CLIP": "internvideo2_clip",
               "InternVideo2_CLIP_small": "internvideo2_clip_small",
               "InternVideo2_Stage2": "internvideo2_stage2"}
    if name not in modules:
        raise AttributeError(name)
    return getattr(import_module("." + modules[name], __name__), name)

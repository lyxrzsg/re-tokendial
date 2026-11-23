"""Use the installed DINOv2 code and weight cache for the official motion entry point."""
import os
import runpy
import torch

hub_load = torch.hub.load

def load(repo_or_dir, model, *args, **kwargs):
    if repo_or_dir == "facebookresearch/dinov2":
        repo_or_dir = os.environ["WAN_DINO_LOCAL"]
        kwargs["source"] = "local"
    return hub_load(repo_or_dir, model, *args, **kwargs)

torch.hub.load = load
runpy.run_module("tokendial.train", run_name="__main__")

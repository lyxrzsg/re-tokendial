import os,runpy
from feature_losses.internvideo2.configs.model import TextEncoders
TextEncoders["bert_large"]["pretrained"]=os.environ["BASELINE_BERT"]
runpy.run_module("tokendial.train",run_name="__main__")

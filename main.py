from PIL import Image
import requests
import torch
import os
import numpy as np
from transformers import AutoProcessor, CLIPVisionModel
from mask import get_mask
from model.pipeline import Pipeline, image_to_audio
from model.aldm import build_audioldm, emb_to_audio
import util
from data.utils import save_wave, emb2seq
import glob

image_to_audio(glob.glob(f'{"images"}/*'), text="", transcription="", save_dir="outputs", config="ssv2a.json",
                gen_remix=True, gen_tracks=False, emb_only=False,
                pretrained="checkpoints/ssv2a.pth")
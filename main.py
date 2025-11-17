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

images = []
labels = []
for path in os.listdir("images"):
  if not path.startswith('.'):
    images.append(np.array(Image.open("images/"+path).convert("RGB")))
    labels.append(path)
print(labels)

for i in range(len(images)):
  img_mask = get_mask(images[i], 200, 200, [1])
  print(img_mask)

  background_mask = images[i].copy()
  for x in range(img_mask.shape[0]):
    for y in range (img_mask.shape[1]):
      if img_mask[x][y] >= 0.5:
        background_mask[x][y] = 0
  Image.fromarray(background_mask).save("images/background_"+labels[i])

  foreground_mask = images[i].copy()
  for x in range(img_mask.shape[0]):
    for y in range (img_mask.shape[1]):
      if img_mask[x][y] < 0.5:
        foreground_mask[x][y] = 0
  Image.fromarray(foreground_mask).save("images/foreground_"+labels[i])

image_to_audio(glob.glob(f'{"images"}/*'), text="", transcription="", save_dir="outputs", config="ssv2a.json",
                gen_remix=True, gen_tracks=False, emb_only=False,
                pretrained="checkpoints/ssv2a.pth")
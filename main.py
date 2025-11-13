from PIL import Image
import requests
import torch
import os
import numpy as np
from transformers import AutoProcessor, CLIPVisionModel
from mask import get_mask
from model.pipeline import Pipeline
from model.aldm import build_audioldm, emb_to_audio
import util
from data.utils import save_wave, emb2seq

model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

images = []
labels = []
for path in os.listdir("images"):
  if not path.startswith('.'):
    images.append(Image.open("images/"+path))
    labels.append(path.split("_")[0])


outputs = np.array([(model(**processor(images=image, return_tensors="pt", output_hidden_states=True)).last_hidden_state[0].detach().numpy()) for image in images])

img_mask = get_mask(images[0], 500, 500, [1])
img_mask = util.downsample_mask_to_patch_weights(torch.tensor([img_mask]), outputs)

background_mask = outputs
for i in range(0, img_mask.shape[1]):
    if img_mask[0][i] >= 0.5:
       background_mask[0][i+1] = np.zeros(outputs.shape[2])

foreground_mask = outputs
for i in range(0, img_mask.shape[1]):
    if img_mask[0][i] < 0.5:
       foreground_mask[0][i+1] = np.zeros(outputs.shape[2])

model = Pipeline("ssv2a.json", "checkpoints/ssv2a.pth", "cuda")
clips = torch.tensor(outputs).to("cuda")
jumps = [len(img) for img in images]

model.image_to_audio([clip[0] for clip in clips], [clip[1:] for clip in clips], jumps)
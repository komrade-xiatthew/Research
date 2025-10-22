from PIL import Image
import requests
import torch
import os
import numpy as np
from transformers import AutoProcessor, CLIPVisionModel
from mask import get_mask
from model.pipeline import Pipeline
import util

model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

images = []
labels = []
for path in os.listdir("/Users/matthewxia/Documents/Research/images"):
  if not path.startswith('.'):
    images.append(Image.open("/Users/matthewxia/Documents/Research/images/"+path))
    labels.append(path.split("_")[0])


outputs = np.array([(model(**processor(images=image, return_tensors="pt", output_hidden_states=True)).last_hidden_state[0].detach().numpy()) for image in images])

print(outputs)

img_mask = get_mask(images[0], 500, 500, [1])
img_mask = util.downsample_mask_to_patch_weights(torch.tensor([img_mask]), outputs)
print(img_mask)


print(outputs.shape)
print(img_mask.shape)

background_mask = outputs
for i in range(0, img_mask.shape[1]):
    if img_mask[0][i] >= 0.5:
       background_mask[0][i+1] = np.zeros(outputs.shape[2])

foreground_mask = outputs
for i in range(0, img_mask.shape[1]):
    if img_mask[0][i] < 0.5:
       foreground_mask[0][i+1] = np.zeros(outputs.shape[2])

print(background_mask)
print(foreground_mask)

ppl = Pipeline("ssv2a.json", "ssv2a.pth", "cuda")
print(ppl.clips2clap(clips=img_mask))


import sys
sys.path.insert(0, './SSV2A')
import argparse
import glob
import json
import socket
import copy


from model.pipeline import Pipeline, image_to_audio
from data.detect import detect
from data.utils import clip_embed_images
from mask import get_mask
from util import downsample_mask_to_patch_weights

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='SSV2A')
#     parser.add_argument('--cfg', type=str, help='Model Config File')
#     parser.add_argument('--ckpt', type=str, default=None, help='Pretrained Checkpoint')
#     parser.add_argument('--image_dir', type=str, default=None, help='Path to the image files')
#     parser.add_argument('--out_dir', type=str, default='./output', help='Path to save the output audios to')
#     parser.add_argument('--bs', type=int, default=64, help='batch size')
#     parser.add_argument('--var_samples', type=int, default=64, help='variational samples')
#     parser.add_argument('--cycle_its', type=int, default=64, help='number of Cycle Mix iterations')
#     parser.add_argument('--cycle_samples', type=int, default=64, help='number of Cycle Mix samples')
#     parser.add_argument('--duration', type=int, default=10, help='generation duration in seconds')
#     parser.add_argument('--seed', type=int, default=42, help='random seed')
#     parser.add_argument('--device', type=str, default='cuda', help='Computation Device')
#     args = parser.parse_args()

images = glob.glob(f'{"images"}/*')
global_clips = clip_embed_images(images, batch_size=64, device=device)

img_mask = get_mask(images[0], 500, 500, [1])
img_mask = downsample_mask_to_patch_weights(torch.tensor([img_mask]), global_clips)


print(global_clips.shape)

bg_clips = copy.deepcopy(global_clips)
for index, clip in enumerate(bg_clips):
    for i in range(0, img_mask.shape[1]):
        if img_mask[0][i] >= 0.5:
            clip[0][i] = np.zeros(clip.shape[2])

fg_clips = copy.deepcopy(global_clips)
for index, clip in enumerate(fg_clips):
    for i in range(0, img_mask.shape[1]):
        if img_mask[0][i] < 0.5:
            clip[0][i] = np.zeros(clip.shape[2])

'''
python infer.py \
--cfg "/home/wguo/Repos/SDV2A/checkpoints/JS-kl00005-best/model.json" \
--ckpt "/home/wguo/Repos/SDV2A/checkpoints/JS-kl00005-best/best_val.pth" \
--image_dir "/home/wguo/Repos/SDV2A/data/samples/images" \
--bs 16
'''
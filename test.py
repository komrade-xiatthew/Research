import sys
sys.path.insert(0, './SSV2A')
import argparse
import glob
import json
import socket


from model.pipeline import Pipeline, image_to_audio
from data.detect import detect

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
config = json.load(open("ssv2a.json", 'r'))
print(socket.gethostname())
print(detect(images, config["detector"], save_dir = "locals", batch_size = 64, device = socket.gethostname()))

'''
python infer.py \
--cfg "/home/wguo/Repos/SDV2A/checkpoints/JS-kl00005-best/model.json" \
--ckpt "/home/wguo/Repos/SDV2A/checkpoints/JS-kl00005-best/best_val.pth" \
--image_dir "/home/wguo/Repos/SDV2A/data/samples/images" \
--bs 16
'''
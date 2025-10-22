import copy
import gc
import json
import os.path
from pathlib import Path
from shutil import rmtree

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import soundfile as sf

from data.detect import detect
from data.tpairs import tpairs2tclips
from data.utils import clip_embed_images, get_timestamp, save_wave, set_seed, emb2seq, batch_extract_frames, \
    prior_embed_texts
from model.aggregator import Aggregator
from model.clap import clap_embed_auds
from model.aldm import build_audioldm, emb_to_audio
from model.generator import Generator
from model.manifold import Manifold
from model.remixer import Remixer


class Pipeline(nn.Module):
    def __init__(self, config, pretrained=None, device='cuda'):
        super().__init__()
        if not isinstance(config, dict):
            with open(config, 'r') as fp:
                config = json.load(fp)
        self.ckpt_path = Path(config['checkpoints'])
        self.config = config
        self.device = device
        config['device'] = device
        self.clip_dim = config['clip_dim']
        self.clap_dim = config['clap_dim']
        self.fold_dim = config['manifold_dim']

        # SDV2A Manifold
        self.manifold = Manifold(**config)
        # if the generator is just a linear operation, ignore modelling
        if config['generator']['arch'] == 'linear':
            self.generator = self.linear_generator
            self.skip_gen_model = True
        else:
            self.generator = Generator(**config)
            self.skip_gen_model = False

        # SDV2A Remixer
        self.remixer = Remixer(**config)

        # if there is any pretrained, load
        if pretrained:
            self.load(pretrained)

        # timestamp
        self.timestamp = get_timestamp()

    def linear_generator(self, fold_clips):
        gen_claps = self.manifold.clap_encoder.model.solve(fold_clips)
        return gen_claps, 0  # add a fake kl loss term to align with other generator models

    def save(self, filepath):
        state = {
            'timestamp': get_timestamp(),
            'manifold_state': self.manifold.state_dict(),
            'generator_state': None if self.skip_gen_model else self.generator.state_dict(),
            'remixer_state': self.remixer.state_dict()
        }
        torch.save(state, filepath)

    def load(self, filepath):
        state = torch.load(filepath, map_location='cpu')
        mia_states = []

        if 'manifold_state' in state:
            self.manifold.load_state_dict(state['manifold_state'])
        else:
            mia_states.append('manifold')

        if 'generator_state' in state and not self.skip_gen_model:
            self.generator.load_state_dict(state['generator_state'])
        else:
            mia_states.append('generator')

        if 'remixer_state' in state:
            self.remixer.load_state_dict(state['remixer_state'])
        else:
            mia_states.append('remixer')

        if len(mia_states) > 0:
            print(f"These states are missing in the model checkpoint supplied:\n"
                  f"{' '.join(mia_states)}\n"
                  f"Inference will be funky if these modules are involved without training!")

        self.timestamp = state['timestamp']

    def __str__(self):
        return (f"SDV2A@{self.timestamp}"
                f"{json.dumps(self.config, sort_keys=True, indent=4)}")
    
    # reconstruct an audio's clap
    def recon_claps(self, claps, var_samples=64):
        fold_claps = self.manifold.fold_claps(claps, var_samples=var_samples)
        gen_claps = self.generator.fold2claps(fold_claps, var_samples=var_samples)
        return gen_claps

    def clips2foldclaps(self, clips, var_samples=64):
        fold_clips = self.manifold.fold_clips(clips, var_samples=var_samples, normalize=False)
        gen_claps = self.generator.fold2claps(fold_clips, var_samples=var_samples)
        return gen_claps

    def clips2folds(self, clips, var_samples=64, normalize=False):
        fold_clips = self.manifold.fold_clips(clips, var_samples=var_samples, normalize=normalize)
        return fold_clips

    def claps2folds(self, claps, var_samples=64, normalize=False):
        fold_claps = self.manifold.fold_claps(claps, var_samples=var_samples, normalize=normalize)
        return fold_claps

    def clips2clap(self, clips, var_samples=64, normalize=False):
        src = self.clips2folds(clips, var_samples=var_samples, normalize=False)
        if self.remixer.guidance == 'generator':
            src = self.generator.fold2claps(src, var_samples=var_samples)
        elif self.remixer.guidance == 'manifold+generator':
            fold_gen_claps = self.generator.fold2claps(src, var_samples=var_samples)
            src = torch.cat([src, fold_gen_claps], dim=-1)
        clap = self.remixer.sample(src, clips, var_samples=var_samples, normalize=normalize)
        return clap


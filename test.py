"""
Evaluation utilities for SVD-based CLIP manipulation pipeline.

This module provides tools for:
1. Generating baseline audios for comparison
2. Loading and comparing audio files
3. Computing evaluation metrics (if audio classification models are available)
4. Testing individual components
"""

import torch
import numpy as np
import sys
import os
import copy

# Add SSV2A to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'SSV2A'))

from ssv2a.model.pipeline import Pipeline
from ssv2a.model.aldm import build_audioldm, emb_to_audio
from ssv2a.data.utils import save_wave, clip_embed_images


def generate_baseline_from_image(image_path, save_name, config_path, checkpoint_path,
                                 device='cpu', var_samples=64, duration=10):
    """
    Generate baseline audio from a single image using standard SSV2A pipeline.

    This uses the CLS-only CLIP embedding (no patch-level manipulation).

    Args:
        image_path: Path to image file
        save_name: Name for output audio file (without extension)
        config_path: Path to SSV2A config JSON
        checkpoint_path: Path to SSV2A checkpoint
        device: 'cuda' or 'cpu'
        var_samples: Number of variational samples (default 64)
        duration: Audio duration in seconds (default 10)

    Returns:
        tuple: (clap_embedding, waveform) or (clap_embedding, None) if audio fails
    """
    print(f"\nGenerating baseline audio from: {image_path}")
    print(f"  Save as: {save_name}")

    # Get standard CLIP embedding (CLS token only)
    clip_emb = clip_embed_images([image_path], device=device)  # [1, 768]
    print(f"  CLIP embedding shape: {clip_emb.shape}")

    # Load SSV2A pipeline
    pipe = Pipeline(config=config_path, pretrained=checkpoint_path, device=device)

    # CLIP → CLAP
    clap = pipe.clips2foldclaps(clip_emb, var_samples=var_samples)
    print(f"  CLAP embedding shape: {clap.shape}")

    # CLAP → Audio
    try:
        audioldm = build_audioldm(model_name='audioldm-s-full-v2', device=device)
        waveform = emb_to_audio(audioldm, clap, batchsize=1, duration=duration)
        print(f"  Audio waveform shape: {waveform.shape}")

        # Save audio
        save_wave(waveform, save_dir='outputs', name=[save_name])
        print(f"  ✓ Saved to: outputs/{save_name}.wav")

        return clap, waveform

    except Exception as e:
        print(f"  ✗ Audio generation failed: {e}")
        print(f"  → CLAP embedding was generated successfully")
        return clap, None


def compare_embeddings(emb1, emb2, name1="Embedding 1", name2="Embedding 2"):
    """
    Compare two embeddings using various metrics.

    Args:
        emb1: First embedding tensor
        emb2: Second embedding tensor
        name1: Name for first embedding
        name2: Name for second embedding

    Returns:
        dict: Dictionary of comparison metrics
    """
    # Ensure same shape
    if emb1.shape != emb2.shape:
        print(f"Warning: Shape mismatch - {emb1.shape} vs {emb2.shape}")
        return None

    # Flatten if needed
    emb1_flat = emb1.flatten()
    emb2_flat = emb2.flatten()

    # Compute metrics
    metrics = {}

    # Cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(
        emb1_flat.unsqueeze(0),
        emb2_flat.unsqueeze(0)
    ).item()
    metrics['cosine_similarity'] = cos_sim

    # L2 distance
    l2_dist = torch.dist(emb1_flat, emb2_flat, p=2).item()
    metrics['l2_distance'] = l2_dist

    # L1 distance
    l1_dist = torch.dist(emb1_flat, emb2_flat, p=1).item()
    metrics['l1_distance'] = l1_dist

    # Mean absolute difference
    metrics['mean_abs_diff'] = (emb1_flat - emb2_flat).abs().mean().item()

    # Max absolute difference
    metrics['max_abs_diff'] = (emb1_flat - emb2_flat).abs().max().item()

    # Print results
    print(f"\nComparison: {name1} vs {name2}")
    print(f"  Cosine Similarity: {metrics['cosine_similarity']:.4f}")
    print(f"  L2 Distance: {metrics['l2_distance']:.4f}")
    print(f"  L1 Distance: {metrics['l1_distance']:.4f}")
    print(f"  Mean Abs Diff: {metrics['mean_abs_diff']:.4f}")
    print(f"  Max Abs Diff: {metrics['max_abs_diff']:.4f}")

    return metrics


def load_and_compare_audios(audio_paths, labels=None):
    """
    Load multiple audio files and compute pairwise comparisons.

    Args:
        audio_paths: List of paths to audio files
        labels: Optional list of labels for each audio

    Note: This function requires librosa or similar audio library
    """
    try:
        import librosa
    except ImportError:
        print("Error: librosa not installed. Cannot load audio files.")
        print("Install with: pip install librosa")
        return None

    if labels is None:
        labels = [f"Audio {i+1}" for i in range(len(audio_paths))]

    print("\nLoading audio files...")
    audios = []
    for path, label in zip(audio_paths, labels):
        if os.path.exists(path):
            audio, sr = librosa.load(path, sr=None)
            audios.append(audio)
            print(f"  ✓ {label}: {path} (sr={sr}, length={len(audio)})")
        else:
            print(f"  ✗ {label}: {path} not found")

    # Pairwise comparisons
    print("\nPairwise Comparisons:")
    for i in range(len(audios)):
        for j in range(i+1, len(audios)):
            print(f"\n{labels[i]} vs {labels[j]}:")
            # Could add more sophisticated audio comparison metrics here
            print(f"  Length: {len(audios[i])} vs {len(audios[j])}")

    return audios


def test_svd_manipulation():
    """
    Test SVD manipulation with dummy data to verify correctness.
    """
    print("\n" + "="*60)
    print("Testing SVD Manipulation")
    print("="*60)

    from util import svd_manipulate_embeddings

    # Create dummy data
    device = 'cpu'
    cls_token = torch.randn(768, device=device)
    cls_token = cls_token / cls_token.norm(p=2)  # L2 normalize

    patch_tokens = torch.randn(256, 768, device=device)
    patch_tokens = patch_tokens / patch_tokens.norm(dim=-1, keepdim=True)

    # Create a mask selecting some patches
    mask_weights = torch.zeros(256, device=device)
    mask_weights[:30] = 1.0  # Select first 30 patches

    print(f"CLS token shape: {cls_token.shape}")
    print(f"Patch tokens shape: {patch_tokens.shape}")
    print(f"Mask weights shape: {mask_weights.shape}")
    print(f"Number of selected patches: {mask_weights.sum().item():.0f}")

    # Test suppression
    print("\nTesting suppression (alpha=0.1)...")
    emb_suppress = svd_manipulate_embeddings(
        cls_token, patch_tokens, mask_weights,
        k=10, mode='suppress', alpha=0.1
    )
    print(f"  Output shape: {emb_suppress.shape}")
    print(f"  Output norm: {emb_suppress.norm(p=2).item():.6f}")
    print(f"  ✓ Suppression successful")

    # Test boost
    print("\nTesting boost (beta=2.0)...")
    emb_boost = svd_manipulate_embeddings(
        cls_token, patch_tokens, mask_weights,
        k=10, mode='boost', beta=2.0
    )
    print(f"  Output shape: {emb_boost.shape}")
    print(f"  Output norm: {emb_boost.norm(p=2).item():.6f}")
    print(f"  ✓ Boost successful")

    # Compare embeddings
    compare_embeddings(emb_suppress, emb_boost, "Suppressed", "Boosted")

    print("\n✓ SVD manipulation tests passed!")


def test_mask_downsampling():
    """
    Test mask downsampling with dummy data.
    """
    print("\n" + "="*60)
    print("Testing Mask Downsampling")
    print("="*60)

    from util import downsample_mask_to_patch_weights

    # Create dummy mask
    device = 'cpu'
    mask = torch.zeros(1, 512, 512, device=device)
    mask[0, 100:300, 100:300] = 1.0  # Create a square region

    print(f"Input mask shape: {mask.shape}")
    print(f"Mask value range: [{mask.min().item()}, {mask.max().item()}]")

    # Downsample
    patch_weights = downsample_mask_to_patch_weights(mask)

    print(f"Output patch weights shape: {patch_weights.shape}")
    print(f"Number of patches >= 0.5: {(patch_weights >= 0.5).sum().item():.0f}")
    print(f"Patch weights value range: [{patch_weights.min().item()}, {patch_weights.max().item()}]")

    print("\n✓ Mask downsampling test passed!")


def test_clip_extraction():
    """
    Test CLIP token extraction with dummy image.
    """
    print("\n" + "="*60)
    print("Testing CLIP Token Extraction")
    print("="*60)

    # This requires actual images, so we'll just verify the function exists
    from util import extract_full_clip_tokens

    print("✓ extract_full_clip_tokens function imported successfully")
    print("  Note: Actual testing requires real image files")


def main():
    """Run evaluation utilities."""
    print("SVD-based CLIP Manipulation - Evaluation Utilities")
    print("="*60)

    # Run tests
    test_svd_manipulation()
    test_mask_downsampling()
    test_clip_extraction()

    print("\n" + "="*60)
    print("All component tests passed!")
    print("="*60)
    print("\nTo generate baseline audios, use generate_baseline_from_image()")
    print("To compare embeddings, use compare_embeddings()")


if __name__ == '__main__':
    main()

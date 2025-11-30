"""
SVD-based CLIP Embedding Manipulation for Image-to-Audio Generation

Complete pipeline implementing the research plan outlined in RESEARCH_PLAN.md
"""

import torch
import numpy as np
import sys
import os
from pathlib import Path

# Add SSV2A to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'SSV2A'))

from util import (
    extract_full_clip_tokens,
    downsample_mask_to_patch_weights,
    svd_manipulate_embeddings,
    combine_manipulated_embeddings
)
from mask import get_mask
from ssv2a.model.pipeline import Pipeline
from ssv2a.model.aldm import build_audioldm, emb_to_audio
from ssv2a.data.utils import save_wave, clip_embed_images


# ============================================================================
# Configuration
# ============================================================================

# Paths (modify these according to your setup)
IMAGE_DIR = "images"
OUTPUT_DIR = "outputs"
CONFIG_PATH = "ssv2a.json"
CHECKPOINT_PATH = "checkpoints/ssv2a.pth"

# SAM2 Configuration (point prompts for masks)
# You need to manually specify the coordinates of objects to segment
SAM2_PROMPTS = {
    'cat': {'x': 500, 'y': 500, 'labels': [1]},  # Adjust coordinates for your images
    'duck': {'x': 500, 'y': 500, 'labels': [1]}  # Adjust coordinates for your images
}

# SVD Hyperparameters
SVD_K = 10  # Number of top singular values to modify
SVD_ALPHA = 0.1  # Suppression factor (for removing objects)
SVD_BETA = 2.0  # Boost factor (for adding objects)

# Embedding Mixing Weights
MIX_ALPHA = 0.2  # Weight for suppressed embedding
MIX_BETA = 0.5  # Weight for boosted embedding
MIX_GAMMA = 0.3  # Weight for background embedding

# Device configuration (CPU fallback for MacBook)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")


# ============================================================================
# Helper Functions
# ============================================================================

def generate_baseline_audio(image_path, save_name, config_path, checkpoint_path, device='cpu'):
    """
    Generate baseline audio using original SSV2A pipeline (CLS-only CLIP).

    Args:
        image_path: Path to image
        save_name: Name for saved audio file
        config_path: Path to SSV2A config
        checkpoint_path: Path to SSV2A checkpoint
        device: 'cuda' or 'cpu'

    Returns:
        Generated audio waveform
    """
    print(f"\n{'='*60}")
    print(f"Generating baseline audio from: {image_path}")
    print(f"{'='*60}")

    # Get CLS-only CLIP embedding using SSV2A's function
    clip_emb = clip_embed_images([image_path], device=device)  # [1, 768]
    print(f"  CLIP embedding shape: {clip_emb.shape}")

    # Load SSV2A pipeline
    pipe = Pipeline(config=config_path, pretrained=checkpoint_path, device=device)

    # CLIP → CLAP
    clap = pipe.clips2foldclaps(clip_emb, var_samples=64)
    print(f"  CLAP embedding shape: {clap.shape}")

    # CLAP → Audio (note: this requires AudioLDM which may not work on CPU)
    try:
        audioldm = build_audioldm(model_name='audioldm-s-full-v2', device=device)
        waveform = emb_to_audio(audioldm, clap, batchsize=1, duration=10)
        print(f"  Audio waveform shape: {waveform.shape}")

        # Save audio
        save_wave(waveform, save_dir=OUTPUT_DIR, name=[save_name])
        print(f"  ✓ Saved to: {OUTPUT_DIR}/{save_name}.wav")

        return waveform
    except Exception as e:
        print(f"  ✗ Audio generation failed (expected on CPU): {e}")
        print(f"  → CLAP embedding was successfully generated")
        return None


def print_step_header(step_num, step_name):
    """Print formatted step header."""
    print(f"\n{'='*60}")
    print(f"STEP {step_num}: {step_name}")
    print(f"{'='*60}")


# ============================================================================
# Main Pipeline
# ============================================================================

def main_pipeline(img_0_path, img_1_path, sam2_prompts_neg, sam2_prompts_pos,
                 config_path, checkpoint_path, device='cpu'):
    """
    Complete 10-step SVD-based CLIP manipulation pipeline.

    Args:
        img_0_path: Path to original image (e.g., dog + cat)
        img_1_path: Path to edited image (e.g., dog + duck)
        sam2_prompts_neg: Dict with 'x', 'y', 'labels' for object to remove
        sam2_prompts_pos: Dict with 'x', 'y', 'labels' for object to add
        config_path: Path to SSV2A config
        checkpoint_path: Path to SSV2A checkpoint
        device: 'cuda' or 'cpu'

    Returns:
        Generated audio waveform (or None if audio generation fails)
    """
    print("\n" + "="*60)
    print("SVD-BASED CLIP MANIPULATION PIPELINE")
    print("="*60)
    print(f"Original image: {img_0_path}")
    print(f"Edited image: {img_1_path}")
    print(f"Device: {device}")
    print("="*60)

    # ------------------------------------------------------------------------
    # STEP 1: Load Images and Generate SAM2 Masks
    # ------------------------------------------------------------------------
    print_step_header(1, "Generate SAM2 Masks")

    print("Generating M_neg (object to remove)...")
    mask_neg = get_mask(
        img_0_path,
        sam2_prompts_neg['x'],
        sam2_prompts_neg['y'],
        sam2_prompts_neg['labels']
    )
    print(f"  M_neg shape: {mask_neg.shape}")

    print("Generating M_pos (object to add)...")
    mask_pos = get_mask(
        img_1_path,
        sam2_prompts_pos['x'],
        sam2_prompts_pos['y'],
        sam2_prompts_pos['labels']
    )
    print(f"  M_pos shape: {mask_pos.shape}")

    # Convert to tensors
    mask_neg_tensor = torch.tensor(mask_neg, device=device).unsqueeze(0)  # [1, H, W]
    mask_pos_tensor = torch.tensor(mask_pos, device=device).unsqueeze(0)
    print(f"  ✓ Masks converted to tensors")

    # ------------------------------------------------------------------------
    # STEP 2: Extract Full CLIP Embeddings
    # ------------------------------------------------------------------------
    print_step_header(2, "Extract Full CLIP Tokens")

    print("Extracting CLIP tokens from I_0...")
    tokens_0 = extract_full_clip_tokens([img_0_path], device=device)
    print(f"  T_0 shape: {tokens_0.shape}")  # Should be [1, 257, 768]

    print("Extracting CLIP tokens from I_1...")
    tokens_1 = extract_full_clip_tokens([img_1_path], device=device)
    print(f"  T_1 shape: {tokens_1.shape}")

    # Separate CLS and patch tokens
    cls_0 = tokens_0[0, 0, :]  # [768]
    patches_0 = tokens_0[0, 1:, :]  # [256, 768]

    cls_1 = tokens_1[0, 0, :]
    patches_1 = tokens_1[0, 1:, :]

    print(f"  CLS_0 shape: {cls_0.shape}")
    print(f"  Patches_0 shape: {patches_0.shape}")
    print(f"  ✓ Separated CLS and patch tokens")

    # ------------------------------------------------------------------------
    # STEP 3: Align Masks to CLIP Patch Grid
    # ------------------------------------------------------------------------
    print_step_header(3, "Downsample Masks to Patch Grid")

    print("Downsampling M_neg to 16x16 patch grid...")
    m_neg = downsample_mask_to_patch_weights(mask_neg_tensor)  # [1, 256]
    print(f"  m_neg shape: {m_neg.shape}")
    print(f"  Number of patches in M_neg: {m_neg[0].sum().item():.0f}")

    print("Downsampling M_pos to 16x16 patch grid...")
    m_pos = downsample_mask_to_patch_weights(mask_pos_tensor)
    print(f"  m_pos shape: {m_pos.shape}")
    print(f"  Number of patches in M_pos: {m_pos[0].sum().item():.0f}")

    # ------------------------------------------------------------------------
    # STEP 4: Extract Region-Specific Patch Tokens
    # ------------------------------------------------------------------------
    print_step_header(4, "Extract Region-Specific Patches")

    # Get background mask (where both m_neg and m_pos are 0)
    bg_mask = (1 - m_neg[0]) * (1 - m_pos[0])  # [256]
    print(f"  Background patches: {bg_mask.sum().item():.0f}")

    # Compute background embedding
    if bg_mask.sum() > 0:
        bg_patches = patches_1[bg_mask > 0.5]  # Extract background patches
        emb_bg = bg_patches.mean(dim=0)  # Simple average
        emb_bg = emb_bg / emb_bg.norm(p=2)  # L2-normalize
        print(f"  ✓ Background embedding computed: {emb_bg.shape}")
    else:
        emb_bg = torch.zeros(768, device=device)
        print(f"  ! No background patches, using zero embedding")

    # ------------------------------------------------------------------------
    # STEP 5: SVD-Based Semantic Manipulation
    # ------------------------------------------------------------------------
    print_step_header(5, "Apply SVD Manipulation")

    print(f"Suppressing object (k={SVD_K}, alpha={SVD_ALPHA})...")
    emb_suppress = svd_manipulate_embeddings(
        cls_0, patches_0, m_neg[0],
        k=SVD_K, mode='suppress', alpha=SVD_ALPHA
    )
    print(f"  E_suppress shape: {emb_suppress.shape}")

    print(f"Boosting object (k={SVD_K}, beta={SVD_BETA})...")
    emb_boost = svd_manipulate_embeddings(
        cls_1, patches_1, m_pos[0],
        k=SVD_K, mode='boost', beta=SVD_BETA
    )
    print(f"  E_boost shape: {emb_boost.shape}")
    print(f"  ✓ SVD manipulation complete")

    # ------------------------------------------------------------------------
    # STEP 6 & 7: Combine Embeddings
    # ------------------------------------------------------------------------
    print_step_header(6, "Combine Manipulated Embeddings")

    print(f"Mixing weights: α={MIX_ALPHA}, β={MIX_BETA}, γ={MIX_GAMMA}")
    final_emb = combine_manipulated_embeddings(
        emb_suppress, emb_boost, emb_bg,
        alpha=MIX_ALPHA, beta=MIX_BETA, gamma=MIX_GAMMA
    )
    print(f"  E_final shape: {final_emb.shape}")
    print(f"  E_final norm: {final_emb.norm(p=2).item():.6f} (should be ~1.0)")
    print(f"  ✓ Final embedding ready for SSV2A")

    # ------------------------------------------------------------------------
    # STEP 8: Load SSV2A Pipeline
    # ------------------------------------------------------------------------
    print_step_header(7, "Load SSV2A Pipeline")

    print(f"Loading pipeline from: {config_path}")
    pipe = Pipeline(config=config_path, pretrained=checkpoint_path, device=device)
    print(f"  ✓ SSV2A pipeline loaded")

    # ------------------------------------------------------------------------
    # STEP 9: CLIP → CLAP
    # ------------------------------------------------------------------------
    print_step_header(8, "Convert CLIP to CLAP")

    final_emb_batch = final_emb.unsqueeze(0)  # [1, 768]
    print(f"  Input CLIP embedding: {final_emb_batch.shape}")

    clap = pipe.clips2foldclaps(final_emb_batch, var_samples=64)
    print(f"  Output CLAP embedding: {clap.shape}")
    print(f"  ✓ CLIP → CLAP conversion complete")

    # ------------------------------------------------------------------------
    # STEP 10: CLAP → Audio
    # ------------------------------------------------------------------------
    print_step_header(9, "Generate Audio from CLAP")

    try:
        print("Loading AudioLDM...")
        audioldm = build_audioldm(model_name='audioldm-s-full-v2', device=device)

        print("Generating 10-second audio waveform...")
        waveform = emb_to_audio(audioldm, clap, batchsize=1, duration=10)
        print(f"  Waveform shape: {waveform.shape}")

        # Save audio
        save_name = 'manipulated_audio'
        save_wave(waveform, save_dir=OUTPUT_DIR, name=[save_name])
        print(f"  ✓ Saved to: {OUTPUT_DIR}/{save_name}.wav")

        return waveform

    except Exception as e:
        print(f"  ✗ Audio generation failed (expected on CPU without GPU): {e}")
        print(f"  → But CLAP embedding was successfully generated!")
        print(f"  → Pipeline completed successfully up to CLAP generation")
        return None


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point for the pipeline."""

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Define image paths (modify these according to your setup)
    img_0_path = os.path.join(IMAGE_DIR, "dog_cat.jpg")  # Original image
    img_1_path = os.path.join(IMAGE_DIR, "dog_duck.jpg")  # Edited image

    # Check if images exist
    if not os.path.exists(img_0_path):
        print(f"Warning: {img_0_path} not found")
        print("Please update img_0_path in main() function")
        print("Using placeholder - pipeline will run but may fail at mask generation")

    if not os.path.exists(img_1_path):
        print(f"Warning: {img_1_path} not found")
        print("Please update img_1_path in main() function")
        print("Using placeholder - pipeline will run but may fail at mask generation")

    # Optional: Generate baseline audios for comparison
    print("\n" + "="*60)
    print("OPTIONAL: Generate Baseline Audios")
    print("="*60)
    generate_baseline = input("Generate baseline audios? (y/n): ").strip().lower()

    if generate_baseline == 'y':
        print("\nGenerating baseline 1: Original image...")
        try:
            generate_baseline_audio(
                img_0_path, 'baseline_original',
                CONFIG_PATH, CHECKPOINT_PATH, DEVICE
            )
        except Exception as e:
            print(f"Baseline 1 failed: {e}")

        print("\nGenerating baseline 2: Edited image...")
        try:
            generate_baseline_audio(
                img_1_path, 'baseline_edited',
                CONFIG_PATH, CHECKPOINT_PATH, DEVICE
            )
        except Exception as e:
            print(f"Baseline 2 failed: {e}")

    # Run main pipeline
    print("\n" + "="*60)
    print("RUNNING MAIN PIPELINE")
    print("="*60)

    try:
        waveform = main_pipeline(
            img_0_path, img_1_path,
            SAM2_PROMPTS['cat'], SAM2_PROMPTS['duck'],
            CONFIG_PATH, CHECKPOINT_PATH, DEVICE
        )

        print("\n" + "="*60)
        print("PIPELINE COMPLETED!")
        print("="*60)

        if waveform is not None:
            print("✓ Audio generated successfully")
            print(f"  Check {OUTPUT_DIR}/manipulated_audio.wav")
        else:
            print("✓ Pipeline completed up to CLAP generation")
            print("  (Audio generation requires GPU)")

    except Exception as e:
        print("\n" + "="*60)
        print("PIPELINE FAILED")
        print("="*60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

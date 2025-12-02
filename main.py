"""
SVD-based CLIP Embedding Manipulation for Image-to-Audio Generation

Complete pipeline implementing the research plan outlined in RESEARCH_PLAN.md
"""

import torch
import numpy as np
import sys
import os
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

from util import (
    extract_full_clip_tokens,
    downsample_mask_to_patch_weights,
    svd_manipulate_embeddings,
    combine_manipulated_embeddings
)
from mask import get_mask
from model.pipeline import Pipeline
from model.aldm import build_audioldm, emb_to_audio
from data.utils import save_wave, clip_embed_images


def visualize_mask(image_path, mask, save_path, title="Mask Visualization", alpha=0.5):
    """
    Visualize SAM2 mask overlaid on the original image.
    
    Args:
        image_path: Path to the original image
        mask: Binary mask numpy array [H, W]
        save_path: Path to save the visualization
        title: Title for the plot
        alpha: Transparency of the mask overlay
    """
    # Load image
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)
    
    # Resize mask to match image if needed
    if mask.shape != (img_array.shape[0], img_array.shape[1]):
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        mask_img = mask_img.resize((img_array.shape[1], img_array.shape[0]), Image.NEAREST)
        mask = np.array(mask_img) / 255.0
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(img_array)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Mask only
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title("SAM2 Mask")
    axes[1].axis('off')
    
    # Overlay: image with colored mask
    overlay = img_array.copy()
    mask_colored = np.zeros_like(img_array)
    mask_colored[:, :, 0] = 255  # Red channel
    mask_colored[:, :, 1] = 50   # Some green
    mask_3d = np.stack([mask, mask, mask], axis=-1)
    overlay = (overlay * (1 - alpha * mask_3d) + mask_colored * alpha * mask_3d).astype(np.uint8)
    
    axes[2].imshow(overlay)
    axes[2].set_title("Mask Overlay")
    axes[2].axis('off')
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved visualization to: {save_path}")


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
    'cat': {'x': 680, 'y': 707, 'labels': [1]},  # Adjust coordinates for your images
    'duck': {'x': 680, 'y': 707, 'labels': [1]}  # Adjust coordinates for your images
}

# SVD Hyperparameters
# With the full SSV2A pipeline (YOLO + Remixer), we can use more meaningful SVD manipulation
SVD_K = 5  # Number of top singular values to modify
SVD_ALPHA = 0.3  # Suppression factor (reduce cat semantics)
SVD_BETA = 1.5  # Boost factor (enhance duck semantics)

# Embedding Mixing Weights for the manipulated global embedding
# This global embedding is injected into SSV2A's Remixer at slot 0
MIX_ALPHA = 0.1  # Small contribution from suppressed embedding
MIX_BETA = 0.4  # Contribution from SVD-boosted embedding
MIX_GAMMA = 0.1  # Small contribution from background
MIX_DIRECT = 0.4  # Base contribution from normal edited image CLIP

# Device configuration (CPU fallback for MacBook)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")


# ============================================================================
# Helper Functions
# ============================================================================

def generate_baseline_audio(image_path, save_name, config_path, checkpoint_path, device='cpu'):
    """
    Generate baseline audio using the FULL original SSV2A pipeline.
    
    This uses SSV2A's image_to_audio() which includes:
    - Object detection (YOLO)
    - Local CLIP embeddings for each detected object
    - Cycle-mix remixing
    
    Args:
        image_path: Path to image
        save_name: Name for saved audio file
        config_path: Path to SSV2A config
        checkpoint_path: Path to SSV2A checkpoint
        device: 'cuda' or 'cpu'

    Returns:
        Generated audio waveform path
    """
    from model.pipeline import image_to_audio
    
    print(f"\n{'='*60}")
    print(f"Generating baseline audio from: {image_path}")
    print(f"Using FULL SSV2A pipeline (detection + remix)")
    print(f"{'='*60}")

    try:
        # Use the full SSV2A image_to_audio pipeline
        image_to_audio(
            images=[image_path],
            text="",
            transcription="",
            save_dir=OUTPUT_DIR,
            config=config_path,
            gen_remix=True,
            gen_tracks=False,
            emb_only=False,
            pretrained=checkpoint_path,
            batch_size=64,
            var_samples=64,
            shuffle_remix=True,
            cycle_its=64,  # Match original SSV2A
            cycle_samples=64,  # Match original SSV2A
            keep_data_cache=False,
            duration=10,
            seed=42,
            device=device
        )
        
        # The output file will be named after the input image
        import os
        output_name = os.path.basename(image_path).replace('.png', '').replace('.jpg', '').replace('.jpeg', '')
        output_path = os.path.join(OUTPUT_DIR, f"{output_name}.wav")
        
        # Rename to the requested save_name
        final_path = os.path.join(OUTPUT_DIR, f"{save_name}.wav")
        if os.path.exists(output_path) and output_path != final_path:
            import shutil
            shutil.move(output_path, final_path)
        
        print(f"  ✓ Saved to: {final_path}")
        return final_path
        
    except Exception as e:
        print(f"  ✗ Audio generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


@torch.no_grad()
def manipulated_image_to_audio(images, manipulated_global_emb, save_name,
                                config_path, checkpoint_path,
                                batch_size=64, var_samples=64,
                                shuffle_remix=True, cycle_its=64, cycle_samples=64,
                                duration=10, seed=42, device='cuda'):
    """
    Generate audio using SSV2A pipeline with a manipulated global embedding.
    
    This preserves YOLO detection for local objects but replaces the global
    CLIP embedding with your SVD-manipulated version.
    
    Args:
        images: List of image paths (for YOLO detection of local objects)
        manipulated_global_emb: Pre-computed manipulated global embedding [B, 768]
        save_name: Name for output audio file
        config_path: Path to SSV2A config JSON
        checkpoint_path: Path to SSV2A checkpoint
        batch_size: Batch size for processing
        var_samples: Number of variational samples
        shuffle_remix: Whether to shuffle in remixer
        cycle_its: Number of cycle iterations
        cycle_samples: Number of cycle samples
        duration: Audio duration in seconds
        seed: Random seed
        device: 'cuda' or 'cpu'
    
    Returns:
        Generated waveform or None if failed
    """
    import copy
    import json
    from pathlib import Path
    from shutil import rmtree
    from data.detect import detect
    from data.utils import clip_embed_images, emb2seq, set_seed, save_wave
    from model.pipeline import Pipeline
    from model.aldm import build_audioldm, emb_to_audio
    
    set_seed(seed)
    
    print(f"\n{'='*60}")
    print("MANIPULATED IMAGE-TO-AUDIO (Hybrid Pipeline)")
    print(f"{'='*60}")
    print(f"  Images: {images}")
    print(f"  Manipulated global embedding shape: {manipulated_global_emb.shape}")
    
    # Load config
    with open(config_path, 'r') as fp:
        config = json.load(fp)
    
    save_dir = Path(OUTPUT_DIR)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    cache_dir = save_dir / 'data_cache'
    
    # Step 1: YOLO detection for local objects (original SSV2A)
    print("\n  [1/5] Running YOLO detection for local objects...")
    local_imgs = detect(images, config['detector'],
                        save_dir=cache_dir / 'masked_images', 
                        batch_size=batch_size, device=device)
    
    # Step 2: Get local CLIP embeddings for detected objects
    print("  [2/5] Extracting local CLIP embeddings...")
    imgs = []
    for img in images:
        imgs += [li for li, _ in local_imgs[img]]
    
    if len(imgs) > 0:
        local_clips = clip_embed_images(imgs, batch_size=batch_size, device=device)
        print(f"       Detected {len(imgs)} local objects")
    else:
        print("       No objects detected, using global embedding only")
        local_clips = manipulated_global_emb.unsqueeze(0) if manipulated_global_emb.dim() == 1 else manipulated_global_emb
    
    jumps = [len(local_imgs[img]) for img in local_imgs]
    
    # Step 3: Load SSV2A pipeline and run remixer
    print("  [3/5] Loading SSV2A pipeline...")
    model = Pipeline(copy.deepcopy(config), checkpoint_path, device)
    model.eval()
    
    print("  [4/5] Running cycle_mix with manipulated global embedding...")
    with torch.no_grad():
        # Build remix_clips: local embeddings + manipulated global at slot 0
        remix_clips = emb2seq(jumps, local_clips, max_length=model.remixer.slot, delay=1, device=model.device)
        
        # KEY: Inject manipulated global embedding at slot 0 (instead of normal global)
        if manipulated_global_emb.dim() == 1:
            manipulated_global_emb = manipulated_global_emb.unsqueeze(0)
        remix_clips[:, 0, :] = manipulated_global_emb.to(device)
        
        print(f"       remix_clips shape: {remix_clips.shape}")
        print(f"       Slot 0 (manipulated global) injected")
        
        # Run cycle_mix remixer
        remix_clap = model.cycle_mix(remix_clips, its=cycle_its, var_samples=var_samples,
                                     samples=cycle_samples, shuffle=shuffle_remix)
        print(f"       remix_clap shape: {remix_clap.shape}")
    
    del remix_clips, local_clips
    
    # Step 4: Generate audio with AudioLDM
    print("  [5/5] Generating audio with AudioLDM...")
    audioldm_v = config['audioldm_version']
    audioldm = build_audioldm(model_name=audioldm_v, device=device)
    waveform = emb_to_audio(audioldm, remix_clap, batchsize=batch_size, duration=duration)
    print(f"       Waveform shape: {waveform.shape}")
    
    # Save audio
    save_wave(waveform, save_dir, name=[save_name])
    output_path = os.path.join(save_dir, f"{save_name}.wav")
    print(f"  ✓ Saved to: {output_path}")
    
    # Cleanup
    if os.path.exists(cache_dir):
        rmtree(cache_dir)
    
    return waveform


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

    # Visualize masks
    print("\nSaving mask visualizations...")
    visualize_mask(
        img_0_path, mask_neg,
        os.path.join(OUTPUT_DIR, "mask_neg_visualization.png"),
        title="M_neg: Object to Remove (Cat)"
    )
    visualize_mask(
        img_1_path, mask_pos,
        os.path.join(OUTPUT_DIR, "mask_pos_visualization.png"),
        title="M_pos: Object to Add (Duck)"
    )

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
    print(f"  T_0 shape: {tokens_0.shape}")

    print("Extracting CLIP tokens from I_1...")
    tokens_1 = extract_full_clip_tokens([img_1_path], device=device)
    print(f"  T_1 shape: {tokens_1.shape}")

    # Separate CLS and patch tokens and move to device
    cls_0 = tokens_0[0, 0, :].to(device)  # [768]
    patches_0 = tokens_0[0, 1:, :].to(device)  # [256, 768]

    cls_1 = tokens_1[0, 0, :].to(device)
    patches_1 = tokens_1[0, 1:, :].to(device)

    print(f"  CLS_0 shape: {cls_0.shape}")
    print(f"  Patches_0 shape: {patches_0.shape}")
    print(f"  ✓ Separated CLS and patch tokens (moved to {device})")

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
    # STEP 6 & 7: Combine Embeddings (with direct CLIP embedding)
    # ------------------------------------------------------------------------
    print_step_header(6, "Combine Embeddings")

    # Get normal CLIP embedding from edited image (this works reliably)
    normal_clip_emb = clip_embed_images([img_1_path], device=device)[0].to(device)
    
    # Mix: mostly normal CLIP + tiny SVD contribution
    print(f"Mixing: {MIX_DIRECT*100:.0f}% normal CLIP + {MIX_BETA*100:.0f}% SVD-boosted")
    
    # Combine with normalized weights
    final_emb = MIX_DIRECT * normal_clip_emb + MIX_BETA * emb_boost.float()
    
    # Add small contributions from suppress/bg if weights > 0
    if MIX_ALPHA > 0:
        final_emb = final_emb + MIX_ALPHA * emb_suppress.float()
    if MIX_GAMMA > 0:
        final_emb = final_emb + MIX_GAMMA * emb_bg.float()
    
    # L2-normalize (critical for SSV2A)
    final_emb = final_emb / final_emb.norm(p=2)
    
    print(f"  E_final shape: {final_emb.shape}")
    print(f"  E_final norm: {final_emb.norm(p=2).item():.6f}")
    print(f"  ✓ Final embedding ready for SSV2A")

    # ------------------------------------------------------------------------
    # STEP 7: Generate Audio with Hybrid Pipeline
    # (YOLO detection + Remixer + Manipulated Global Embedding)
    # ------------------------------------------------------------------------
    print_step_header(7, "Generate Audio with Hybrid Pipeline")
    
    print("Using manipulated global embedding with full SSV2A pipeline:")
    print("  - YOLO detection for local object embeddings")
    print("  - Your SVD-manipulated embedding as global (slot 0)")
    print("  - Remixer (cycle_mix) for audio generation")
    
    try:
        # Use the edited image for YOLO detection (it has the duck we want)
        # The manipulated global embedding encodes the semantic changes
        waveform = manipulated_image_to_audio(
            images=[img_1_path],  # Use edited image for YOLO detection
            manipulated_global_emb=final_emb.float(),
            save_name='manipulated_audio',
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            batch_size=64,
            var_samples=64,
            shuffle_remix=True,
            cycle_its=64,  # Match original SSV2A
            cycle_samples=64,  # Match original SSV2A
            duration=10,
            seed=42,
            device=device
        )
        
        return waveform

    except Exception as e:
        print(f"  ✗ Audio generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point for the pipeline."""

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Define image paths (modify these according to your setup)
    img_0_path = os.path.join(IMAGE_DIR, "original.JPG")  # Original image
    img_1_path = os.path.join(IMAGE_DIR, "edit.JPG")  # Edited image

    # Check if images exist
    if not os.path.exists(img_0_path):
        print(f"Warning: {img_0_path} not found")
        print("Please update img_0_path in main() function")
        print("Using placeholder - pipeline will run but may fail at mask generation")

    if not os.path.exists(img_1_path):
        print(f"Warning: {img_1_path} not found")
        print("Please update img_1_path in main() function")
        print("Using placeholder - pipeline will run but may fail at mask generation")

    # Optional: Generate baseline audio for comparison
    print("\n" + "="*60)
    print("OPTIONAL: Generate Baseline Audio (Original Image)")
    print("="*60)
    generate_baseline = input("Generate baseline audio? (y/n): ").strip().lower()

    if generate_baseline == 'y':
        print("\nGenerating baseline: Original image...")
        try:
            generate_baseline_audio(
                img_0_path, 'baseline_original',
                CONFIG_PATH, CHECKPOINT_PATH, DEVICE
            )
        except Exception as e:
            print(f"Baseline generation failed: {e}")

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
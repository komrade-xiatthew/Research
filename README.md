# SVD-based CLIP Embedding Manipulation for Image-to-Audio Generation

Implementation of novel research combining image segmentation (SAM2), visual semantics (CLIP), and audio generation (SSV2A) using Singular Value Decomposition for semantic manipulation.

## Overview

This project implements a pipeline that:
1. Takes an original image and an edited image
2. Uses SAM2 to segment objects that were removed/added
3. Extracts full CLIP patch tokens (not just CLS token)
4. Applies SVD-based suppression/amplification to region-specific embeddings
5. Feeds manipulated CLIP embeddings through SSV2A to generate audio

**Research Question:** Can we generate audio that reflects visual edits by manipulating CLIP embeddings at the patch level using SVD?

## Files

- **`RESEARCH_PLAN.md`**: Complete detailed research methodology and mathematical foundation
- **`main.py`**: Complete 10-step pipeline implementation
- **`util.py`**: Core utility functions (CLIP extraction, SVD manipulation, embedding combination)
- **`mask.py`**: SAM2 mask generation
- **`test.py`**: Evaluation utilities and component tests
- **`ssv2a.json`**: SSV2A model configuration

## Installation

### Prerequisites

```bash
# Python 3.8+
pip install torch torchvision
pip install clip-by-openai
pip install pillow numpy

# For SAM2 (if not already installed)
pip install git+https://github.com/facebookresearch/sam2.git

# For audio processing (optional)
pip install librosa
```

### Setup

1. Ensure SSV2A repository is in parent directory:
   ```
   Matt_research/
   ├── SSV2A/
   └── Research/  (this repo)
   ```

2. Download SAM2 checkpoint:
   ```bash
   mkdir -p checkpoints
   # Place sam2.1_hiera_large.pt in checkpoints/
   ```

3. Download SSV2A checkpoint:
   ```bash
   # Place ssv2a.pth in checkpoints/
   ```

## Usage

### Quick Start

```bash
# Run component tests (no GPU required)
python test.py

# Run main pipeline (requires images and checkpoints)
python main.py
```

### Configuration

Edit `main.py` to configure:

```python
# Image paths
img_0_path = "images/dog_cat.jpg"  # Original image
img_1_path = "images/dog_duck.jpg"  # Edited image

# SAM2 prompts (adjust coordinates for your images)
SAM2_PROMPTS = {
    'cat': {'x': 500, 'y': 500, 'labels': [1]},
    'duck': {'x': 500, 'y': 500, 'labels': [1]}
}

# SVD Hyperparameters
SVD_K = 10  # Number of top singular values to modify
SVD_ALPHA = 0.1  # Suppression factor
SVD_BETA = 2.0  # Boost factor

# Mixing Weights
MIX_ALPHA = 0.2  # Suppressed embedding
MIX_BETA = 0.5  # Boosted embedding
MIX_GAMMA = 0.3  # Background embedding
```

### Step-by-Step Usage

#### 1. Prepare Images

Place your images in the `images/` directory:
- Original image (e.g., dog + cat)
- Edited image (e.g., dog + duck)

#### 2. Identify Object Coordinates

Open your images and note the (x, y) coordinates of:
- Object to remove (e.g., cat in original image)
- Object to add (e.g., duck in edited image)

Update `SAM2_PROMPTS` in `main.py` with these coordinates.

#### 3. Run Pipeline

```bash
python main.py
```

The pipeline will:
1. Generate SAM2 masks
2. Extract CLIP tokens (257 tokens per image)
3. Downsample masks to 16×16 patch grid
4. Apply SVD manipulation
5. Combine embeddings
6. Generate CLAP embeddings (requires SSV2A checkpoint)
7. Generate audio (requires GPU for AudioLDM)

#### 4. Check Outputs

Results will be saved in `outputs/`:
- `manipulated_audio.wav`: Audio from manipulated CLIP embeddings
- `baseline_original.wav`: Baseline audio from original image (optional)
- `baseline_edited.wav`: Baseline audio from edited image (optional)

## Key Functions

### `util.py`

#### `extract_full_clip_tokens(images, device='cpu')`
Extract all 257 CLIP tokens (1 CLS + 256 patches) instead of just CLS.

```python
tokens = extract_full_clip_tokens(['image.jpg'], device='cpu')
# Returns: [1, 257, 768] tensor
```

#### `downsample_mask_to_patch_weights(mask, patch_grid_size=16)`
Align SAM2 masks to CLIP's 16×16 patch grid.

```python
mask = torch.tensor(sam2_mask).unsqueeze(0)  # [1, H, W]
patch_weights = downsample_mask_to_patch_weights(mask)
# Returns: [1, 256] binary weights
```

#### `svd_manipulate_embeddings(cls_token, patch_tokens, mask_weights, k=10, mode='suppress', alpha=0.1, beta=2.0)`
Apply SVD-based semantic manipulation.

```python
# Suppress cat semantics
emb_suppress = svd_manipulate_embeddings(
    cls_0, patches_0, m_neg,
    k=10, mode='suppress', alpha=0.1
)

# Boost duck semantics
emb_boost = svd_manipulate_embeddings(
    cls_1, patches_1, m_pos,
    k=10, mode='boost', beta=2.0
)
# Returns: [768] L2-normalized embedding
```

#### `combine_manipulated_embeddings(emb_suppress, emb_boost, emb_bg, alpha=0.2, beta=0.5, gamma=0.3)`
Combine multiple manipulated embeddings.

```python
final_emb = combine_manipulated_embeddings(
    emb_suppress, emb_boost, emb_bg,
    alpha=0.2, beta=0.5, gamma=0.3
)
# Returns: [768] L2-normalized combined embedding
```

### `mask.py`

#### `get_mask(image_path, point_x, point_y, point_labels)`
Generate segmentation mask using SAM2.

```python
mask = get_mask('image.jpg', x=500, y=500, labels=[1])
# Returns: [H, W] binary numpy array
```

## Architecture Details

### CLIP Token Structure (ViT-L/14 @ 224×224)
- **Total tokens**: 257
  - 1 CLS token (global image representation)
  - 256 patch tokens (16×16 grid, each patch is 14×14 pixels)
- **Embedding dimension**: 768
- **Output shape**: `[batch, 257, 768]`

### SVD Manipulation Process

1. **Select region patches** using downsampled SAM2 mask
2. **Stack tokens**: `X = [CLS; selected_patches]`
3. **SVD decomposition**: `X = U × Σ × V^T`
4. **Modify singular values**:
   - Suppression: `Σ'[0:k] = Σ[0:k] × 0.1`
   - Boost: `Σ'[0:k] = Σ[0:k] × 2.0`
5. **Reconstruct**: `X' = U × Σ' × V^T`
6. **Weighted average** to get final embedding
7. **L2-normalize** for compatibility

### Pipeline Flow

```
Images → SAM2 → Masks [H, W]
             ↓
          Downsample
             ↓
        Patch masks [16×16]
             ↓
Images → CLIP → Tokens [257, 768]
             ↓
      Separate patches [256, 768]
             ↓
    Index by mask + SVD manipulation
             ↓
    Modified embeddings [768]
             ↓
         Combine [768]
             ↓
    SSV2A → CLAP [512]
             ↓
    AudioLDM → Audio
```

## Hyperparameter Tuning

### SVD Parameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `K` | 10 | 5-15 | Number of singular values to modify |
| `alpha` | 0.1 | 0.05-0.3 | Suppression strength (lower = stronger) |
| `beta` | 2.0 | 1.5-3.0 | Boost strength (higher = stronger) |

### Mixing Weights

| Parameter | Default | Range | Purpose |
|-----------|---------|-------|---------|
| `MIX_ALPHA` | 0.2 | 0.1-0.3 | Suppressed embedding weight |
| `MIX_BETA` | 0.5 | 0.4-0.6 | Boosted embedding weight |
| `MIX_GAMMA` | 0.3 | 0.2-0.4 | Background embedding weight |

**Note**: Weights should sum to 1.0.

## Troubleshooting

### Common Issues

1. **"device" not defined error**
   - The code uses CPU by default on MacBook
   - Check `DEVICE` variable in `main.py`

2. **CLIP model download fails**
   - Requires internet connection
   - Model will be cached after first download

3. **SAM2 mask generation fails**
   - Check image path is correct
   - Verify SAM2 checkpoint is in `checkpoints/`
   - Adjust point coordinates for your images

4. **Audio generation fails on CPU**
   - Expected behavior - AudioLDM requires GPU
   - Pipeline will still generate CLAP embeddings
   - Run on GPU for full audio generation

5. **Shape mismatch errors**
   - Ensure using ViT-L/14 CLIP model (not ViT-B/32)
   - Check mask dimensions match image dimensions

## Testing

Run component tests:

```bash
python test.py
```

This will test:
- SVD manipulation with dummy data
- Mask downsampling correctness
- Function imports

## Evaluation

Compare results:

```python
from test import compare_embeddings, generate_baseline_from_image

# Generate baselines
generate_baseline_from_image('image.jpg', 'baseline', 'ssv2a.json', 'checkpoints/ssv2a.pth')

# Compare embeddings
metrics = compare_embeddings(emb1, emb2, "Original", "Manipulated")
```

## Research Contributions

1. **Novel approach**: First work applying SVD to CLIP patch tokens for audio generation
2. **Region-aware manipulation**: Spatial control over audio generation using segmentation masks
3. **Transfer learning**: Adapting AudioEditor's EOT suppression to image-to-audio domain

## Limitations

1. **Generation, not editing**: Creates new audio rather than editing existing audio
2. **Out-of-distribution**: Manipulated embeddings may be outside training manifold
3. **No temporal control**: Cannot specify when objects appear in audio timeline
4. **Requires GPU**: Full pipeline needs GPU for audio generation

## Future Work

- [ ] Integrate with AudioEditor for true audio editing
- [ ] Learn optimal K, α, β values through training
- [ ] Multi-object manipulation (>2 objects)
- [ ] Temporal control for longer sequences
- [ ] Quantitative evaluation metrics

## References

- **SSV2A**: Image-to-Audio generation pipeline
- **AudioEditor**: Diffusion-based audio editing with EOT suppression
- **SAM2**: Segment Anything Model 2 for image segmentation
- **CLIP**: Contrastive Language-Image Pre-training (ViT-L/14)

## License

Research code for academic purposes.

## Contact

For questions about this implementation, refer to `RESEARCH_PLAN.md` for detailed methodology.

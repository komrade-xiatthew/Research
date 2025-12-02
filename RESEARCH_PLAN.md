# SVD-based CLIP Embedding Manipulation for Image-to-Audio Generation

## Research Objective

**Task:** Given original image I_0 and edited image I_1, generate audio that reflects the visual edits through SVD-based manipulation of CLIP embeddings.

**Example:**
- **Input:** I_0 (dog + cat), I_1 (dog + duck)
- **Output:** Generated audio (dog barking + duck quacking)

**Note:** This is **conditional audio generation** (not editing), as we're generating new audio from manipulated visual features rather than editing original audio waveforms.

---

## Mathematical Foundation

### Core Insight
CLIP embeddings encode visual semantics in a high-dimensional space where:
- Different semantic concepts occupy different subspaces
- Singular Value Decomposition (SVD) reveals the principal components of these semantics
- Top singular values capture the most dominant semantic information
- By suppressing/amplifying these values, we can weaken/strengthen specific semantic concepts

### Why SVD Manipulation Works

1. **Semantic Decomposition:** When you stack tokens from a specific region (e.g., cat patches), the top-K singular values represent the strongest semantic signals of that region

2. **Suppression Effect:** Attenuating top-K singular values reduces the "strength" of that semantic concept in the embedding space

3. **Amplification Effect:** Boosting top-K singular values enhances the semantic concept

4. **Preservation:** Lower singular values (noise/details) remain relatively unchanged to maintain structure

### Why This Transfers to Audio

SSV2A learns a mapping from visual semantics (CLIP space) to audio semantics (CLAP space). If visual semantics are modified (cat suppressed, duck boosted), the learned mapping should produce corresponding audio changes (less cat sounds, more duck sounds).

### Checkpoints
Download sam2.1_hiera_large.pt, ssv2a.pth and yolo8x-oiv7.pt to ./checkpoints

---

## Detailed Step-by-Step Pipeline

### STEP 1: Image Loading and SAM2 Mask Generation

#### 1.1 Load Images
- Load original image I_0 (contains dog + cat)
- Load edited image I_1 (contains dog + duck)
- Format: RGB images, any resolution (will be resized to 224×224 for CLIP)

#### 1.2 Generate SAM2 Masks
- Apply SAM2 to I_0 with point prompt on cat region → **M_neg**
- Apply SAM2 to I_1 with point prompt on duck region → **M_pos**
- Format: Binary numpy arrays [H, W] where 1 = target region, 0 = other

**Why this works:** SAM2 provides object-level segmentation that separates semantic regions. By identifying exactly which pixels correspond to cat vs duck, we can later identify which CLIP patch tokens encode cat semantics vs duck semantics.

---

### STEP 2: Extract Full CLIP Embeddings

#### 2.1 Process Images Through CLIP
- Load CLIP ViT-L/14 model (same as SSV2A uses)
- Pass I_0 through CLIP visual encoder → Extract **all tokens** T_0 [1, 257, 768]
- Pass I_1 through CLIP visual encoder → Extract **all tokens** T_1 [1, 257, 768]
- **Critical:** Extract from the last transformer layer BEFORE pooling

**Architecture details (ViT-L/14):**
- Input image: 224×224 pixels
- Patch size: 14×14 pixels
- Number of patches: (224÷14)² = 16×16 = 256 patches
- Each patch: 768-dimensional embedding
- Plus 1 CLS token = 257 total tokens

#### 2.2 Separate CLS and Patch Tokens
```
From T_0 [1, 257, 768]:
  - cls_0 = T_0[0, 0, :] → [768]
  - patches_0 = T_0[0, 1:, :] → [256, 768]

From T_1 [1, 257, 768]:
  - cls_1 = T_1[0, 0, :] → [768]
  - patches_1 = T_1[0, 1:, :] → [256, 768]
```

**Why:** CLS token captures global context, patch tokens capture local region information.

---

### STEP 3: Align Masks to CLIP Patch Grid

#### 3.1 Resize Masks
- Resize M_neg from [H, W] to [224, 224] using bilinear interpolation
- Resize M_pos from [H, W] to [224, 224] using bilinear interpolation

#### 3.2 Downsample to Patch Grid
- Apply average pooling with kernel_size=14, stride=14
- Result: m_neg [16, 16] and m_pos [16, 16]
- Each value represents what fraction of that 14×14 patch is inside the mask

#### 3.3 Binarize Patch Masks
- Apply threshold: if value >= 0.5 → 1.0, else → 0.0
- Flatten to vectors: m_neg → [256], m_pos → [256]

**Why this works:** Creates perfect spatial alignment between pixel-level masks (SAM2) and patch-level tokens (CLIP).

---

### STEP 4: Extract Region-Specific Patch Tokens

```
From patches_0 [256, 768] and m_neg [256]:
  - Get indices where m_neg == 1 → idx_neg
  - Extract T_neg = patches_0[idx_neg] → [n_neg, 768]

From patches_1 [256, 768] and m_pos [256]:
  - Get indices where m_pos == 1 → idx_pos
  - Extract T_pos = patches_1[idx_pos] → [n_pos, 768]

Background patches:
  - Get indices where (m_neg == 0) AND (m_pos == 0) → idx_bg
  - Extract T_bg = patches_1[idx_bg] → [n_bg, 768]
```

**Result:**
- T_neg: Only patches encoding "cat" semantics
- T_pos: Only patches encoding "duck" semantics
- T_bg: Background/unchanged regions

---

### STEP 5: SVD-Based Semantic Manipulation

#### 5.1 Build Token Matrices

**For SUPPRESSION (remove cat):**
```
X_neg = [cls_0; T_neg]
Shape: [(1 + n_neg), 768]
```

**For AMPLIFICATION (boost duck):**
```
X_pos = [cls_1; T_pos]
Shape: [(1 + n_pos), 768]
```

#### 5.2 Perform SVD Decomposition

```
X_neg = U × Σ × V^T
X_pos = U × Σ × V^T

Where:
- U: Left singular vectors
- Σ: Singular values (diagonal)
- V^T: Right singular vectors (transposed)
```

#### 5.3 Modify Singular Values

**For SUPPRESSION:**
```
Choose K (e.g., K=10)
Σ_new[0:K] = Σ[0:K] × α  where α=0.1 (attenuate)
Σ_new[K:] = Σ[K:]  (preserve)
```

**For AMPLIFICATION:**
```
Choose K (e.g., K=10)
Σ_new[0:K] = Σ[0:K] × β  where β=2.0 (amplify)
Σ_new[K:] = Σ[K:]  (preserve)
```

**Why this works:**
- Top-K singular values represent dominant semantic components
- Suppression (α=0.1): Weakens "cat-ness" in embeddings
- Amplification (β=2.0): Strengthens "duck-ness" in embeddings
- Like adjusting volume on specific semantic channels

**Hyperparameter choices:**
- K: 5-15 (sweet spot for semantic manipulation)
- α: 0.1-0.3 (suppression factor)
- β: 1.5-3.0 (boost factor)

#### 5.4 Reconstruct Modified Matrices

```
X'_neg = U × Σ_new × V^T  (suppressed cat semantics)
X'_pos = U × Σ_new × V^T  (boosted duck semantics)
```

---

### STEP 6: Aggregate to Single Embeddings

#### 6.1 Weighted Averaging

**For suppression embedding:**
```
weights_neg = [1.0; m_neg[idx_neg]]  (CLS + patch weights)
E_suppress = Σ(X'_neg[i] × weights_neg[i]) / Σ(weights_neg[i])
```

**For boost embedding:**
```
weights_pos = [1.0; m_pos[idx_pos]]
E_boost = Σ(X'_pos[i] × weights_pos[i]) / Σ(weights_pos[i])
```

**For background:**
```
E_bg = mean(T_bg, dim=0)
```

#### 6.2 L2-Normalization (CRITICAL)

```
E_suppress = E_suppress / ||E_suppress||_2
E_boost = E_boost / ||E_boost||_2
E_bg = E_bg / ||E_bg||_2
```

**Why critical:** CLIP embeddings are L2-normalized by design. SSV2A's manifold encoder expects unit-normalized inputs.

---

### STEP 7: Combine Embeddings

#### 7.1 Weighted Linear Combination

```
E_final = α × E_suppress + β × E_boost + γ × E_bg

Where α + β + γ = 1.0

Example:
  α = 0.2  (20% suppressed cat)
  β = 0.5  (50% boosted duck)
  γ = 0.3  (30% background)
```

**Intuition:** Creating a "semantic recipe" that steers the audio generation toward desired content.

#### 7.2 Final L2-Normalization

```
E_final = E_final / ||E_final||_2
```

**Result:** [768] L2-normalized manipulated CLIP embedding ready for SSV2A!

---

### STEP 8: Audio Generation via Hybrid SSV2A Pipeline

**Key Innovation:** We preserve SSV2A's full pipeline (YOLO detection + Remixer) but inject our SVD-manipulated embedding as the global context.

#### 8.1 YOLO Detection (preserved from SSV2A)
```
I_1 (edited image) → YOLO detector → detected objects
detected objects → CLIP → local_clips [N, 768]
```
This captures object-level semantics that SSV2A relies on for quality audio.

#### 8.2 Inject Manipulated Global Embedding
```
E_final [768] → remix_clips[:, 0, :] = E_final  (slot 0 = global context)
local_clips → remix_clips[:, 1:, :]  (slots 1+ = detected objects)
```
**Your SVD-manipulated embedding replaces the normal global CLIP embedding.**

#### 8.3 Remixer (cycle_mix)
```
remix_clips [B, slots, 768] → cycle_mix() → remix_clap [B, 512]
```
The Remixer combines global context with local object embeddings to produce a coherent CLAP embedding.

#### 8.4 CLAP → Audio Waveform
```
remix_clap [512] → AudioLDM.emb_to_audio() → waveform [1, 220500]
```

**Result:** 10-second audio waveform at 22.05kHz

**Why this works better:**
- Preserves SSV2A's object detection for local audio sources
- Your SVD manipulation affects the global "scene context"
- Remixer normalizes and blends embeddings, reducing OOD issues

---

### STEP 9: Evaluation & Comparison

#### Baselines

**NOTE:** SSV2A's full pipeline is more complex than just CLS embedding:
1. Object detection (YOLO) → segments objects in image
2. Local CLIP embeddings for each detected object
3. Global CLIP embedding for full image  
4. Remixer with cycle_mix to combine local+global embeddings
5. AudioLDM to generate audio

**Baseline: Original image (full SSV2A pipeline)**
```
I_0 → YOLO detect → local CLIPs + global CLIP → Remixer → AudioLDM → audio_original
Expected: Dog + cat sounds
```

#### Your Method (Hybrid Approach)
```
Step 1: Generate manipulated global embedding
  I_0 + I_1 + SAM2 masks → SVD manipulation → E_final [768]

Step 2: Run SSV2A with manipulated global
  I_1 → YOLO detect → local_clips
  E_final → remix_clips[:, 0, :]  (inject as global)
  local_clips → remix_clips[:, 1:, :]
  remix_clips → cycle_mix → remix_clap → AudioLDM → audio_manipulated

Expected: Dog + duck sounds (with semantic control via SVD)
```

#### Metrics

**Qualitative:**
- Listen and compare presence/strength of cat vs duck sounds
- Check if audio_manipulated is closer to audio_edited than audio_original

**Quantitative:**
- Use audio classifier (VGGish, PANN)
- Measure confidence scores for "cat", "dog", "duck"
- Compare scores across all three audios

**Expected results:**
- Higher duck score than audio_original
- Lower cat score than audio_original
- Similar dog score (unchanged region)

---

## Implementation Files

### util.py Functions

1. **extract_full_clip_tokens(images, batch_size, device)**
   - Input: List of image paths
   - Output: [B, 257, 768] tensor of all CLIP tokens

2. **downsample_mask_to_patch_weights(mask, patch_grid_size=16)**
   - Input: [B, H, W] binary mask
   - Output: [B, 256] patch-aligned weights

3. **svd_manipulate_embeddings(cls_token, patch_tokens, mask_weights, k, mode, alpha, beta)**
   - Input: CLS token, patches, mask, hyperparameters
   - Output: [768] manipulated embedding

4. **combine_manipulated_embeddings(emb_suppress, emb_boost, emb_bg, alpha, beta, gamma)**
   - Input: Three embeddings + mixing weights
   - Output: [768] combined embedding

### mask.py Functions

1. **get_mask(image_path, point_x, point_y, point_labels)**
   - Input: Image path and SAM2 point prompt
   - Output: [H, W] binary mask

### main.py Pipeline

**Hybrid approach** combining SVD manipulation with full SSV2A:

1. Load images and generate SAM2 masks
2. Extract CLIP tokens (full 257 tokens, not just CLS)
3. Downsample masks to patch grid (256 patches)
4. Extract region-specific patches
5. Apply SVD manipulation (suppress cat, boost duck)
6. Combine embeddings to create manipulated global
7. **Call manipulated_image_to_audio():**
   - YOLO detection on edited image → local object embeddings
   - Inject manipulated global at slot 0
   - Run cycle_mix Remixer
   - AudioLDM generates audio
8. Save results

---

## Research Contribution

**Novel aspects:**
1. First work to apply SVD-based semantic manipulation to CLIP embeddings for audio generation
2. Region-aware audio generation using spatial masks (SAM2)
3. Transfer of AudioEditor's EOT suppression concept to image-to-audio domain
4. **Hybrid pipeline** that preserves SSV2A's audio quality while enabling semantic control

**Key insight:**
SVD manipulation alone creates out-of-distribution embeddings. By integrating with SSV2A's full pipeline (YOLO + Remixer), the manipulated global embedding is normalized and blended with detected object embeddings, producing higher quality audio.

**Motivation:**
We propose SVD-based manipulation as a global semantic prior that guides SSV2A's generation. While local object detection stills contributes, the global embedding provides high-level semantic control over the scene's audio characteristics.

---

## Hyperparameters Summary

| Parameter | Default | Range | Purpose |
|-----------|---------|-------|---------|
| K | 5 | 1-15 | Number of singular values to modify |
| α (suppress) | 0.3 | 0.1-0.5 | Suppression factor for removal |
| β (boost) | 1.5 | 1.1-3.0 | Amplification factor for addition |
| α_mix (suppress weight) | 0.1 | 0.0-0.3 | Mixing weight for suppressed embedding |
| β_mix (boost weight) | 0.4 | 0.2-0.6 | Mixing weight for boosted embedding |
| γ_mix (bg weight) | 0.1 | 0.0-0.3 | Mixing weight for background |
| direct_mix | 0.4 | 0.2-0.6 | Weight for normal edited image CLIP |
| cycle_its | 64 | - | SSV2A cycle mix iterations |
| cycle_samples | 64 | - | SSV2A cycle mix samples |
| Patch grid size | 16 | - | For ViT-L/14 @ 224×224 |
| Mask threshold | 0.5 | 0.4-0.6 | Binarization threshold |

---

## Code Changes from Original SSV2A

This section documents modifications made to implement the SVD-based manipulation pipeline.

### Modified Files

#### 1. main.py (New File)
Main entry point implementing the hybrid SVD + SSV2A pipeline.

**New Functions:**
- `visualize_mask()`: Saves SAM2 mask visualizations
- `generate_baseline_audio()`: Generates baseline audio using full SSV2A pipeline
- `manipulated_image_to_audio()`: **Key function** - runs SSV2A with manipulated global embedding
- `main_pipeline()`: Orchestrates the full SVD manipulation pipeline
- `set_seed()`: Sets random seed for reproducibility

**Key Parameters:**
```python
# SVD Hyperparameters
SVD_K = 5           # Number of top singular values to modify
SVD_ALPHA = 0.3     # Suppression factor
SVD_BETA = 1.5      # Boost factor

# Embedding Mixing Weights
MIX_ALPHA = 0.1     # Suppressed embedding weight
MIX_BETA = 0.4      # SVD-boosted embedding weight
MIX_GAMMA = 0.1     # Background embedding weight
MIX_DIRECT = 0.4    # Normal CLIP embedding weight

# SSV2A Parameters (matching original)
cycle_its = 64      # Cycle mix iterations
cycle_samples = 64  # Cycle mix samples
var_samples = 64    # Variational samples
```

#### 2. util.py (New File)
Utility functions for CLIP token extraction and SVD manipulation.

**Functions:**
- `extract_full_clip_tokens()`: Extracts all 257 CLIP tokens (CLS + 256 patches)
- `downsample_mask_to_patch_weights()`: Aligns SAM2 masks to CLIP patch grid
- `svd_manipulate_embeddings()`: Core SVD-based semantic manipulation
- `combine_manipulated_embeddings()`: Combines suppressed/boosted/background embeddings

#### 3. mask.py (New File)
SAM2 mask generation wrapper.

**Functions:**
- `get_mask()`: Generates binary mask using SAM2 with point prompts

#### 4. model/pipeline.py (Unchanged from SSV2A)
Original SSV2A pipeline preserved intact. The `image_to_audio()` function is used as-is for baseline generation.

#### 5. verify_masks.py (New File)
Interactive tool to verify SAM2 mask coordinates before running the main pipeline.

### How the Hybrid Pipeline Works

```
1. SAM2 masks identify regions (cat to remove, duck to add)
2. CLIP tokens extracted from both images
3. SVD manipulation creates modified global embedding:
   - Suppress cat semantics in original image patches
   - Boost duck semantics in edited image patches
   - Combine with background and direct CLIP embedding
4. manipulated_image_to_audio() runs SSV2A with:
   - YOLO detection → local object CLIP embeddings (unchanged)
   - Manipulated global embedding injected at slot 0 (your contribution)
   - cycle_mix Remixer blends local + global (unchanged)
   - AudioLDM generates final audio (unchanged)
```

---

**End of Research Plan**
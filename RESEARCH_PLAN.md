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

### STEP 8: Audio Generation via SSV2A Pipeline

#### 8.1 CLIP → Manifold Encoding
```
E_final [768] → pipe.manifold.fold_clips() → manifold_emb [768]
```

#### 8.2 Manifold → CLAP Generation
```
manifold_emb [768] → pipe.generator.fold2claps() → clap_emb [512]
```

**This is where visual→audio transfer happens!** The generator learned to map visual features to audio features.

#### 8.3 CLAP → Audio Waveform
```
clap_emb [512] → AudioLDM.emb_to_audio() → waveform [1, 220500]
```

**Result:** 10-second audio waveform at 22.05kHz

---

### STEP 9: Evaluation & Comparison

#### Baselines

**Baseline 1: Original image**
```
I_0 → CLIP (CLS only) → SSV2A → audio_original
Expected: Dog + cat sounds
```

**Baseline 2: Edited image**
```
I_1 → CLIP (CLS only) → SSV2A → audio_edited
Expected: Dog + duck sounds
```

#### Your Method
```
I_0 + I_1 + masks → manipulated CLIP → SSV2A → audio_manipulated
Expected: Dog + duck sounds (with modified strength)
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

Complete 10-step orchestration:
1. Load images and generate masks
2. Extract CLIP tokens
3. Downsample masks to patch grid
4. Extract region-specific patches
5. Apply SVD manipulation
6. Aggregate to single embeddings
7. Combine embeddings
8. Load SSV2A pipeline
9. CLIP → CLAP → Audio
10. Save results

---

## Research Contribution

**Novel aspects:**
1. First work to apply SVD-based semantic manipulation to CLIP embeddings for audio generation
2. Region-aware audio generation using spatial masks
3. Transfer of AudioEditor's EOT suppression concept to image-to-audio domain

**Limitations:**
1. This is generation, not editing (doesn't preserve original audio)
2. SVD-manipulated embeddings are out-of-distribution
3. Success depends on SSV2A's learned visual-audio mapping

**Future work:**
1. Integrate with AudioEditor for true audio editing
2. Learn optimal K, α, β values
3. Multi-object manipulation
4. Temporal control for longer audio sequences

---

## Hyperparameters Summary

| Parameter | Default | Range | Purpose |
|-----------|---------|-------|---------|
| K | 10 | 5-15 | Number of singular values to modify |
| α (suppress) | 0.1 | 0.1-0.3 | Suppression factor for removal |
| β (boost) | 2.0 | 1.5-3.0 | Amplification factor for addition |
| α_mix (suppress weight) | 0.2 | 0.1-0.3 | Mixing weight for suppressed embedding |
| β_mix (boost weight) | 0.5 | 0.4-0.6 | Mixing weight for boosted embedding |
| γ_mix (bg weight) | 0.3 | 0.2-0.4 | Mixing weight for background |
| Patch grid size | 16 | - | For ViT-L/14 @ 224×224 |
| Mask threshold | 0.5 | 0.4-0.6 | Binarization threshold |

---

**End of Research Plan**

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

## Implementation Guide

**Starting Point:** You will begin from the `main` branch, which contains the basic SSV2A pipeline. Your task is to implement the SVD-based CLIP manipulation approach by adding new functions and rewriting the pipeline orchestration.

**What exists in main branch:**
- `util.py`: Basic utility with `downsample_mask_to_patch_weights()` function (needs modification)
- `mask.py`: SAM2 integration with `get_masks()` batch function
- `main.py`: Simple 16-line script calling SSV2A's `image_to_audio()`
- `model/pipeline.py`: SSV2A pipeline (with some experimental code to remove)

**What you need to implement:**
- `util.py` (3 new functions + 1 refactor)
- `mask.py` (1 new function)
- `main.py` (complete rewrite)
- `verify_masks.py` (new file)
- Documentation and cleanup

**Estimated time:** 20-25 hours of focused implementation

---

## Implementation Roadmap

### Priority 1: Core Algorithm Functions (util.py) - 4-6 hours

The main branch's `util.py` contains only 55 lines. You need to expand it to ~295 lines by adding three critical functions and refactoring one existing function.

---

#### Task 1.1: Add `extract_full_clip_tokens()` function (~75 lines)

**What exists:** Nothing - this function doesn't exist in main branch

**What you need to add:**
A function that extracts ALL 257 CLIP tokens (1 CLS + 256 patches), not just the pooled CLS token.

**Function signature:**
```python
def extract_full_clip_tokens(images, batch_size=64, device='cuda'):
    """Returns: [N, 257, 768] tensor"""
```

**Implementation requirements:**

1. **Load CLIP ViT-L/14 model**
   - Use `clip.load('ViT-L/14', device=device)`
   - Store both model and preprocessing function

2. **Process images in batches** for memory efficiency
   - Iterate through images in chunks of `batch_size`
   - Load each image with PIL, apply preprocessing
   - Stack into batch tensor

3. **Manually traverse visual encoder (KEY CHALLENGE)**
   - **Problem:** Standard `model.encode_image()` only returns [B, 768] CLS token
   - **What you need:** Access intermediate layers to get all 257 tokens
   - **Hint:** Look at how CLIP's visual encoder works:
     - `model.visual.conv1()` - convolution layer
     - Add CLS token and positional embeddings
     - `model.visual.transformer()` - transformer layers
     - `model.visual.ln_post()` - layer norm
     - `model.visual.proj` - projection matrix
   - **Critical:** Apply projection to ALL tokens, not just CLS

4. **Handle dtype correctly**
   - CLIP uses float16 on GPU
   - Convert batch to model's dtype: `batch_tensors.to(model.dtype)`

5. **L2-normalize result**
   - Each token should be unit-normalized
   - `result / result.norm(dim=-1, keepdim=True)`

**Expected output:**
- Shape: `[N, 257, 768]`
- Position 0: CLS token (global)
- Positions 1-256: Patch tokens (16×16 spatial grid)
- All tokens L2-normalized

**Why this matters:** Without patch tokens, you cannot perform spatial SVD manipulation. CLS token loses spatial information.

---

#### Task 1.2: Add `svd_manipulate_embeddings()` function (~85 lines) ⭐ MOST CRITICAL

**What exists:** Nothing - this is your core innovation

**What you need to add:**
The SVD-based semantic manipulation algorithm that suppresses or amplifies region-specific semantics.

**Function signature:**
```python
def svd_manipulate_embeddings(cls_token, patch_tokens, mask_weights,
                               k=10, mode='suppress', alpha=0.1, beta=2.0):
    """Returns: [768] L2-normalized embedding"""
```

**Implementation requirements:**

1. **Extract masked patches**
   - Find indices where `mask_weights > 0.5`
   - Handle edge case: what if no patches selected? (return zeros or raise error)
   - Extract: `selected_patches = patch_tokens[mask_indices]`

2. **Build SVD matrix**
   - Stack `[CLS_token; selected_patches]` into matrix X
   - Shape: `[1 + n_selected, 768]`
   - Why include CLS? Provides global context for the region

3. **Handle dtype for SVD**
   - PyTorch SVD doesn't support float16
   - Save original dtype, convert to float32
   - You'll convert back before returning

4. **Perform SVD decomposition**
   - `U, S, Vt = torch.linalg.svd(X, full_matrices=False)`
   - X = U × diag(S) × Vt
   - S contains singular values in descending order

5. **Modify top-K singular values** ⭐ **THIS IS THE CORE TRICK**
   - Clone S (don't modify in-place!)
   - If `mode='suppress'`: `S_modified[:k] = S[:k] * alpha` (e.g., α=0.1 → reduce to 10%)
   - If `mode='boost'`: `S_modified[:k] = S[:k] * beta` (e.g., β=2.0 → amplify to 200%)
   - Leave S[k:] unchanged (preserve fine details)
   - **Why this works:** Top-K singular values represent dominant semantic components
     - Suppressing them weakens "cat-ness"
     - Boosting them strengthens "duck-ness"

6. **Reconstruct modified matrix**
   - `X_modified = U @ torch.diag(S_modified) @ Vt`
   - Now you have semantically altered token representations

7. **Aggregate to single embedding**
   - Compute weighted average of all rows in X_modified
   - Weight CLS as 1.0, weight patches by their mask values
   - Normalize weights to sum to 1.0

8. **L2-normalize and convert dtype**
   - Normalize: `embedding / embedding.norm(p=2)`
   - Convert back to original dtype
   - Return `[768]` vector

**Parameters to understand:**
- **k**: Number of singular values to modify (5-15 typical range)
  - Too low: Doesn't capture enough semantics
  - Too high: Affects fine details, unstable
- **alpha** (suppress): 0.1 = strong, 0.5 = mild, 1.0 = no effect
- **beta** (boost): 1.0 = no effect, 2.0 = moderate, 3.0 = strong

**Common pitfalls:**
- Forgetting float32 conversion → SVD will crash
- Modifying S in-place → corrupts original
- Not L2-normalizing → breaks CLIP compatibility
- Not handling empty masks → crashes on edge cases

---

#### Task 1.3: Add `combine_manipulated_embeddings()` function (~35 lines)

**What exists:** Nothing - this function doesn't exist

**What you need to add:**
Combine multiple manipulated embeddings (suppressed, boosted, background) into final embedding.

**Function signature:**
```python
def combine_manipulated_embeddings(emb_suppress, emb_boost, emb_bg,
                                   alpha=0.2, beta=0.5, gamma=0.3):
    """Returns: [768] L2-normalized combined embedding"""
```

**Implementation requirements:**

1. **Validate weights**
   - Check: `abs(alpha + beta + gamma - 1.0) < 0.01`?
   - If not, print warning (but don't fail)
   - Why? Weights should sum to 1.0 to preserve magnitude

2. **Linear combination**
   - `combined = alpha * emb_suppress + beta * emb_boost + gamma * emb_bg`
   - This creates a "semantic recipe"
   - Example: 20% suppressed cat + 50% boosted duck + 30% background

3. **L2-normalize**
   - `combined / combined.norm(p=2)`
   - Why? Linear combination may not preserve unit norm

**Expected output:** `[768]` unit-normalized embedding

**Parameter meanings:**
- **alpha**: Weight for suppressed region (0.1-0.3 typical)
- **beta**: Weight for boosted region (0.4-0.6 typical)
- **gamma**: Weight for background (0.2-0.4 typical)
- Experiment with different combinations!

---

#### Task 1.4: Refactor `downsample_mask_to_patch_weights()`

**What exists in main:** Function with signature:
```python
downsample_mask_to_patch_weights(mask, emb, target_res=224)
```

**What you need to change:**

1. **Remove** the `emb` parameter (no longer needed)
2. **Add** two new parameters:
   - `patch_grid_size=16` (for ViT-L/14's 16×16 grid)
   - `threshold=0.5` (for binarization)
3. **Simplify** the logic:
   - Resize mask to 224×224 using `F.interpolate()`
   - Downsample to patch grid using `F.avg_pool2d(kernel_size=14, stride=14)`
   - Flatten to `[B, 256]`
   - Binarize: `>= threshold → 1.0, else → 0.0`

**New signature:**
```python
def downsample_mask_to_patch_weights(mask, patch_grid_size=16, threshold=0.5):
    """Returns: [B, 256] binary weights"""
```

**Why this matters:** Creates spatial alignment between SAM2 masks and CLIP patches.

---

### Priority 2: Pipeline Orchestration (main.py) - 6-8 hours

The main branch's `main.py` is only 16 lines - a simple call to `image_to_audio()`. You need to completely rewrite it to ~605 lines with configuration, helper functions, and the full 7-step pipeline.

---

#### Task 2.1: Add configuration section (~30 lines)

**What to add:**

1. **Constants for paths:**
   ```python
   IMAGE_DIR = "images"
   OUTPUT_DIR = "outputs"
   CONFIG_PATH = "ssv2a.json"
   CHECKPOINT_PATH = "checkpoints/ssv2a.pth"
   ```

2. **SAM2 prompts dictionary:**
   ```python
   SAM2_PROMPTS = {
       'cat': {'x': 680, 'y': 707, 'labels': [1]},  # Object to remove
       'duck': {'x': 680, 'y': 707, 'labels': [1]}  # Object to add
   }
   ```

3. **SVD hyperparameters:**
   ```python
   SVD_K = 5          # Number of singular values to modify
   SVD_ALPHA = 0.3    # Suppression factor
   SVD_BETA = 1.5     # Boost factor
   ```

4. **Mixing weights:**
   ```python
   MIX_ALPHA = 0.1    # Suppressed embedding
   MIX_BETA = 0.4     # Boosted embedding
   MIX_GAMMA = 0.1    # Background
   MIX_DIRECT = 0.4   # Normal CLIP embedding
   ```

**Why:** Makes hyperparameters easy to tune without changing code.

---

#### Task 2.2: Add helper functions (~150 lines total)

**Helper 1: `visualize_mask()` (~50 lines)**

Purpose: Save visualization of SAM2 mask overlaid on image

Requirements:
- Load original image
- Create 3-panel figure: (1) original, (2) mask, (3) overlay
- Save to outputs/ directory
- Useful for debugging mask quality

---

**Helper 2: `generate_baseline_audio()` (~70 lines)**

Purpose: Generate audio using standard SSV2A pipeline (for comparison)

Requirements:
- Load SSV2A Pipeline
- Run full SSV2A: image → YOLO detect → CLIP embeddings → Remixer → CLAP → AudioLDM
- Save audio to outputs/
- **Why:** Establishes baseline to compare against your manipulated version

---

**Helper 3: `manipulated_image_to_audio()` (~120 lines)** ⭐ KEY FUNCTION

Purpose: Run SSV2A with YOUR manipulated global embedding (hybrid approach)

Requirements:
1. **YOLO detection** (preserve SSV2A's object detection)
   - Detect objects in edited image
   - Extract local CLIP embeddings for each object

2. **Inject manipulated global embedding**
   - Your SVD-manipulated embedding goes to slot 0 (global context)
   - YOLO object embeddings go to slots 1+ (local objects)

3. **Remixer (cycle_mix)**
   - Combine global + local embeddings
   - This is SSV2A's original Remixer - don't modify it

4. **CLAP → AudioLDM**
   - Convert to CLAP embedding
   - Generate audio waveform
   - Save to outputs/

**Why this is hybrid:** You keep SSV2A's full pipeline (quality) but inject your semantic control (SVD manipulation).

---

#### Task 2.3: Implement `main_pipeline()` (~200 lines) ⭐ CORE ORCHESTRATION

Purpose: Orchestrate the complete 7-step SVD manipulation pipeline

**STEP 1: Generate SAM2 Masks** (~20 lines)
- Call `get_mask()` for original image → M_neg (cat to remove)
- Call `get_mask()` for edited image → M_pos (duck to add)
- Call `visualize_mask()` for both (save to outputs/)
- Convert numpy arrays to PyTorch tensors
- Add batch dimension: `[H, W] → [1, H, W]`

**STEP 2: Extract Full CLIP Tokens** (~15 lines)
- Call `extract_full_clip_tokens([img_0_path])` → tokens_0 `[1, 257, 768]`
- Call `extract_full_clip_tokens([img_1_path])` → tokens_1 `[1, 257, 768]`
- Separate each into:
  - `cls_0 = tokens_0[0, 0, :]` → `[768]`
  - `patches_0 = tokens_0[0, 1:, :]` → `[256, 768]`
  - Same for tokens_1

**STEP 3: Align Masks to CLIP Patch Grid** (~10 lines)
- Call `downsample_mask_to_patch_weights(M_neg)` → m_neg `[1, 256]`
- Call `downsample_mask_to_patch_weights(M_pos)` → m_pos `[1, 256]`
- Remove batch dimension for easier indexing

**STEP 4: Compute Background Embedding** (~15 lines)
- Compute background mask: `bg_mask = (1 - m_neg) * (1 - m_pos)`
- Extract background patches: `bg_patches = patches_1[bg_mask[0] > 0.5]`
- Average: `emb_bg = bg_patches.mean(dim=0)`
- L2-normalize: `emb_bg = emb_bg / emb_bg.norm(p=2)`

**STEP 5: Apply SVD Manipulation** (~20 lines)
- Suppress:
  ```python
  emb_suppress = svd_manipulate_embeddings(
      cls_0, patches_0, m_neg[0],
      k=SVD_K, mode='suppress', alpha=SVD_ALPHA
  )
  ```
- Boost:
  ```python
  emb_boost = svd_manipulate_embeddings(
      cls_1, patches_1, m_pos[0],
      k=SVD_K, mode='boost', beta=SVD_BETA
  )
  ```

**STEP 6: Get Normal CLIP Embedding + Combine All** (~25 lines)
- Get standard CLIP CLS from edited image (for mixing)
- Combine four components:
  ```python
  final_emb = (
      MIX_DIRECT * normal_clip +
      MIX_BETA * emb_boost +
      MIX_ALPHA * emb_suppress +
      MIX_GAMMA * emb_bg
  )
  ```
- L2-normalize final_emb

**STEP 7: Generate Audio via Hybrid Pipeline** (~30 lines)
- Call `manipulated_image_to_audio()` with:
  - Edited image path (for YOLO detection)
  - Your manipulated global embedding
  - SSV2A config and checkpoint
- Save audio to outputs/

**Additional requirements:**
- Add print statements for each step (progress feedback)
- Handle device placement (CPU vs GPU)
- Include error handling for missing files
- Print shapes at each step for debugging

---

#### Task 2.4: Add `main()` entry point (~70 lines)

Requirements:
1. Create outputs/ directory if not exists
2. Define image paths (original.JPG, edit.JPG)
3. Optional: Generate baseline audio for comparison
4. Call `main_pipeline()` with all parameters
5. Print completion message with output locations

---

### Priority 3: Helper Functions (mask.py, verify_masks.py) - 3-4 hours

#### Task 3.1: Add `get_mask()` to mask.py (~35 lines)

**What exists:** Only `get_masks()` batch function

**What to add:** Single-image convenience function

**Function signature:**
```python
def get_mask(image_path, point_x, point_y, point_labels):
    """Returns: [H, W] binary numpy array"""
```

**Key difference from `get_masks()`:**
- Takes single image path (not list)
- Uses `multimask_output=False` (returns single best mask)
- Returns single mask (not list)

**Implementation:**
1. Load image from path
2. Set in SAM2 predictor
3. Create point prompt: `np.array([[x, y]])`
4. Call `predictor.predict()` with `multimask_output=False`
5. Return `masks[0].astype(np.float32)`

---

#### Task 3.2: Create verify_masks.py (~170 lines) - NEW FILE

Purpose: Interactive tool to find correct SAM2 coordinates before running main pipeline

**Required functions:**

1. **`visualize_point_and_mask()`** (~60 lines)
   - Load image
   - Generate SAM2 mask at given (x, y)
   - Create 3-panel visualization:
     - Panel 1: Image with red star at click point
     - Panel 2: Mask (grayscale)
     - Panel 3: Mask overlay (red tint on image)
   - Save to outputs/

2. **`interactive_mode()`** (~50 lines)
   - Prompt user for image path
   - Prompt for (x, y) coordinates
   - Call visualize_point_and_mask()
   - Ask if user wants to try again
   - Loop until satisfied

3. **`main()`** (~40 lines)
   - Parse command-line arguments (--image, --x, --y)
   - If args provided: one-shot mode
   - If no args: interactive mode

**Usage:**
```bash
# Interactive mode
python verify_masks.py

# CLI mode
python verify_masks.py --image images/original.JPG --x 680 --y 707
```

**Why this matters:** Getting SAM2 coordinates right is critical. This tool lets you verify before running the full pipeline.

---

### Priority 4: Documentation & Cleanup

#### Task 4.1: Create .gitignore (~10 minutes)

Add file to exclude:
- Python cache (`__pycache__/`, `*.pyc`)
- Model checkpoints (`checkpoints/`, `*.pth`, `*.pt`)
- Output files (`outputs/`, `*.wav`)
- Generated visualizations (`*.png`, `*.jpg` except `images/`)
- IDE files (`.vscode/`, `.DS_Store`)
- Virtual environments (`venv/`, `env/`)

Refer to Research/.gitignore template if needed.

---

#### Task 4.2: Write README.md

Sections to include:
1. **Overview** - Research question and approach
2. **Installation** - Dependencies and setup
3. **Usage** - How to run the code
4. **Key Functions** - API documentation
5. **Architecture Details** - CLIP tokens, SVD algorithm, pipeline flow
6. **Hyperparameter Tuning** - Parameter ranges and effects
7. **Troubleshooting** - Common issues and solutions

**Tip:** Focus on making it clear for someone new to the project. Include:
- Example commands
- Expected outputs
- Shape information
- Common pitfalls

---

#### Task 4.3: Cleanup model/pipeline.py

**What to remove:**
- Lines 228-252 in main branch contain experimental code:
  ```python
  from util import downsample_mask_to_patch_weights
  from mask import get_masks
  # ... experimental masking code ...
  ```

**Why remove:** This was a test implementation. The proper implementation is now in main.py.

**Result:** Clean pipeline that matches original SSV2A behavior.

---

### Full Pipeline Test

Run with simple test images:
```bash
python main.py
```

Check:
- [ ] No errors or crashes
- [ ] Mask visualizations saved to outputs/
- [ ] All print statements show correct shapes
- [ ] Final audio file generated (if GPU available)
- [ ] Audio sounds different from baseline

---

## Common Implementation Challenges & Solutions

**Challenge 1: "No module named 'ssv2a'" error**
- **Cause:** Incorrect import path
- **Solution:** Use `from model.pipeline import Pipeline` not `from ssv2a.model...`

**Challenge 2: CLIP token extraction only gives [B, 768]**
- **Cause:** Using `model.encode_image()` directly
- **Solution:** Manually call `model.visual` layers to get intermediate tokens

**Challenge 3: "SVD not implemented for Half" error**
- **Cause:** PyTorch SVD doesn't support float16
- **Solution:** Convert to float32 before SVD, convert back after

**Challenge 4: Device mismatch errors**
- **Cause:** Tensors on different devices (CPU vs GPU)
- **Solution:** Use `.to(device)` consistently, or work entirely on CPU

**Challenge 5: Mask selects wrong patches**
- **Cause:** Incorrect grid size or threshold
- **Solution:** Use `verify_masks.py` to check mask quality, adjust coordinates

**Challenge 6: Out-of-memory on GPU**
- **Cause:** Large batch sizes or multiple models loaded
- **Solution:** Use `batch_size=1`, clear cache between steps

---

## Success Criteria Checklist

Your implementation is complete when:

1. ✅ **All functions implemented**
   - util.py has 4 functions totaling ~295 lines
   - mask.py has `get_mask()` function
   - main.py has full pipeline (~605 lines)
   - verify_masks.py exists and works

2. ✅ **Shapes are correct at every step**
   - CLIP tokens: [1, 257, 768]
   - Downsampled masks: [1, 256]
   - All embeddings: [768]
   - Final audio: [1, 220500] or similar

3. ✅ **Normalization preserved**
   - All embeddings have L2-norm ≈ 1.0
   - No NaN or Inf values anywhere

4. ✅ **SVD manipulation works**
   - Both 'suppress' and 'boost' modes run without errors
   - Handles edge cases (empty masks, single patch)

5. ✅ **Pipeline produces audio**
   - No crashes during execution
   - Mask visualizations saved to outputs/
   - Audio file generated (even if quality unclear)

6. ✅ **Code is clean**
   - Hyperparameters in configuration section
   - Print statements for debugging
   - Error handling for missing files
   - Comments explaining key steps

**When all checkboxes are complete, you're done!**

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
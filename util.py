import torch
import torch.nn.functional as F
import clip
from PIL import Image
import sys
import os

# Add SSV2A to path for compatibility
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'SSV2A'))


def extract_full_clip_tokens(images, batch_size=64, device='cuda'):
    """
    Extract full CLIP patch tokens + CLS token from images.

    Args:
        images: List of image file paths
        batch_size: Batch size for processing
        device: 'cuda' or 'cpu'

    Returns:
        Tensor of shape [N, 257, 768] for ViT-L/14
        - Position 0: CLS token
        - Positions 1-256: Patch tokens (16x16 grid)
    """
    # Load CLIP model
    model, preprocess = clip.load('ViT-L/14', device=device)
    model.eval()

    all_tokens = []

    for i in range(0, len(images), batch_size):
        batch_paths = images[i:i+batch_size]
        batch_imgs = []

        for img_path in batch_paths:
            img = Image.open(img_path).convert('RGB')
            batch_imgs.append(preprocess(img))

        batch_tensors = torch.stack(batch_imgs).to(device)
        # Convert to model's dtype (CLIP uses half precision on GPU)
        batch_tensors = batch_tensors.to(model.dtype)

        with torch.no_grad():
            # Extract features from visual encoder
            # We need to manually go through the visual encoder to get all tokens
            x = model.visual.conv1(batch_tensors)  # shape: [B, hidden_dim, grid, grid]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # [B, hidden_dim, grid*grid]
            x = x.permute(0, 2, 1)  # [B, grid*grid, hidden_dim]

            # Add CLS token
            cls_token = model.visual.class_embedding.to(x.dtype) + torch.zeros(
                x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
            )
            x = torch.cat([cls_token, x], dim=1)  # [B, 1+grid*grid, hidden_dim]

            # Add positional embedding
            x = x + model.visual.positional_embedding.to(x.dtype)

            # Apply pre-LayerNorm
            x = model.visual.ln_pre(x)

            # Pass through transformer
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = model.visual.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD

            # Apply post-LayerNorm
            x = model.visual.ln_post(x)

            # Apply projection to all tokens (not just CLS)
            # model.visual.proj is [1024, 768] for ViT-L/14
            if model.visual.proj is not None:
                x = x @ model.visual.proj

            # x now has shape [B, 257, 768] for ViT-L/14 @ 224x224
            # Position 0 is CLS token, positions 1-256 are patch tokens (16x16 grid)

        all_tokens.append(x.cpu())

    result = torch.cat(all_tokens, dim=0)  # [N, 257, 768]

    # L2-normalize each token (CLIP standard)
    result = result / result.norm(dim=-1, keepdim=True)

    return result


def _compute_resize_and_center_crop(h, w, target=224):
    if h <= w:
        new_h = target
        new_w = int(round(w * target / h))
        top = 0
        left = (new_w - target) // 2
    else:
        new_w = target
        new_h = int(round(h * target / w))
        left = 0
        top = (new_h - target) // 2
    return (new_h, new_w), (top, top + target, left, left + target)

def _apply_resize_crop_mask(mask, new_size, crop_box):
    # mask: B x H x W, values 0 or 1
    x = mask.unsqueeze(1).float()                         # B x 1 x H x W
    x = F.interpolate(x, size=new_size, mode="nearest")   # keep labels
    t, b, l, r = crop_box
    x = x[:, :, t:b, l:r]                                 # B x 1 x 224 x 224
    return x[:, 0]                                        # B x 224 x 224

def downsample_mask_to_patch_weights(mask, patch_grid_size=16, threshold=0.5):
    """
    Downsample SAM2 mask to align with CLIP patch grid.

    Args:
        mask: [B, H, W] binary mask from SAM2 in {0, 1}
        patch_grid_size: 16 for ViT-L/14 @ 224x224 (produces 16x16=256 patches)
        threshold: Binarization threshold (default 0.5)

    Returns:
        [B, 256] binary weights aligned to CLIP patch tokens
        Each value corresponds to one of the 256 patches
    """
    # Ensure mask is a tensor
    if not isinstance(mask, torch.Tensor):
        mask = torch.tensor(mask)

    # Add batch dimension if needed
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)  # [H, W] -> [1, H, W]

    B, H, W = mask.shape

    # Resize mask to 224x224 (CLIP's input size)
    mask_resized = F.interpolate(
        mask.unsqueeze(1).float(),  # [B, 1, H, W]
        size=(224, 224),
        mode='bilinear',
        align_corners=False
    )  # [B, 1, 224, 224]

    # Downsample to patch grid using average pooling
    # Each patch is 14x14 pixels (224 / 16 = 14)
    patch_size = 224 // patch_grid_size
    patch_weights = F.avg_pool2d(
        mask_resized,
        kernel_size=patch_size,
        stride=patch_size
    )  # [B, 1, 16, 16]

    # Flatten to [B, 256]
    patch_weights = patch_weights.view(B, -1)  # [B, 256]

    # Binarize: >= threshold → 1, else → 0
    patch_weights = (patch_weights >= threshold).float()

    return patch_weights  # [B, 256]

def masked_pool(emb, weights, eps=1e-6):
    """
    emb: B x T x D with CLS at 0
    weights: B x (T-1) for patch tokens
    returns: B x D normalized region embedding
    """
    x = emb[:, 1:, :]                 # B x (T-1) x D
    w = weights[:, :, None]           # B x (T-1) x 1
    num = (x * w).sum(dim=1)          # B x D
    den = w.sum(dim=1).clamp_min(eps) # B x 1
    out = num / den
    out = out / out.norm(dim=-1, keepdim=True).clamp_min(eps)
    return out


def svd_manipulate_embeddings(cls_token, patch_tokens, mask_weights, k=10,
                              mode='suppress', alpha=0.1, beta=2.0):
    """
    Apply SVD-based suppression/amplification to region-specific tokens.

    This is the core innovation: using SVD to manipulate semantic content
    in CLIP embedding space.

    Args:
        cls_token: [768] CLS token from CLIP
        patch_tokens: [256, 768] all patch tokens from CLIP
        mask_weights: [256] binary mask indicating which patches belong to target region
        k: number of top singular values to modify (default 10)
        mode: 'suppress' (attenuate) or 'boost' (amplify)
        alpha: suppression factor for mode='suppress' (default 0.1)
        beta: boost factor for mode='boost' (default 2.0)

    Returns:
        [768] L2-normalized embedding after SVD manipulation

    Algorithm:
        1. Extract patches where mask_weights == 1
        2. Stack [CLS; selected_patches] into matrix X
        3. Perform SVD: X = U × Σ × V^T
        4. Modify top-K singular values based on mode
        5. Reconstruct: X' = U × Σ' × V^T
        6. Weighted average of all rows in X'
        7. L2-normalize result
    """
    # Get indices where mask == 1
    mask_indices = (mask_weights > 0.5).nonzero(as_tuple=True)[0]

    # Handle edge case: no patches in this region
    if len(mask_indices) == 0:
        print(f"Warning: No patches selected for {mode} (all mask weights <= 0.5)")
        return torch.zeros(768, device=cls_token.device, dtype=cls_token.dtype)

    # Extract selected patches
    selected_patches = patch_tokens[mask_indices]  # [n, 768]

    # Stack CLS token + selected patches
    X = torch.cat([cls_token.unsqueeze(0), selected_patches], dim=0)  # [n+1, 768]
    
    # Store original dtype and convert to float32 for SVD (not supported for half precision)
    original_dtype = X.dtype
    X = X.float()

    # Perform SVD decomposition
    # X = U × Σ × V^T
    # U: [n+1, n+1], S: [n+1], Vt: [768, 768]
    U, S, Vt = torch.linalg.svd(X, full_matrices=False)

    # Modify top-K singular values
    S_modified = S.clone()
    k_actual = min(k, len(S))  # Don't exceed available singular values

    if mode == 'suppress':
        # Attenuate top-K: reduce strength of dominant semantic components
        S_modified[:k_actual] = S[:k_actual] * alpha
    elif mode == 'boost':
        # Amplify top-K: increase strength of dominant semantic components
        S_modified[:k_actual] = S[:k_actual] * beta
    else:
        raise ValueError(f"Unknown mode: {mode}. Must be 'suppress' or 'boost'")

    # Reconstruct with modified singular values
    # X' = U × diag(Σ') × V^T
    X_modified = U @ torch.diag(S_modified) @ Vt  # [n+1, 768]

    # Weighted average of all tokens in modified matrix
    # Weight CLS token as 1.0, weight patches by their mask values
    weights = torch.ones(len(mask_indices) + 1, device=X.device)
    weights[0] = 1.0  # CLS token weight
    weights[1:] = mask_weights[mask_indices]  # Patch weights

    # Normalize weights
    weights = weights / weights.sum()

    # Compute weighted average
    embedding = (X_modified * weights.unsqueeze(1)).sum(dim=0)  # [768]

    # L2-normalize (critical for CLIP compatibility)
    embedding = embedding / embedding.norm(p=2)

    # Convert back to original dtype
    return embedding.to(original_dtype)  # [768]


def combine_manipulated_embeddings(emb_suppress, emb_boost, emb_bg,
                                   alpha=0.2, beta=0.5, gamma=0.3):
    """
    Combine suppression, boost, and background embeddings.

    Creates a "semantic recipe" by mixing:
    - Suppressed embedding (object being removed, e.g., cat)
    - Boosted embedding (object being added, e.g., duck)
    - Background embedding (unchanged regions)

    Args:
        emb_suppress: [768] embedding with suppressed semantics
        emb_boost: [768] embedding with boosted semantics
        emb_bg: [768] background embedding
        alpha: weight for suppressed embedding (default 0.2)
        beta: weight for boosted embedding (default 0.5)
        gamma: weight for background embedding (default 0.3)

    Returns:
        [768] L2-normalized combined embedding

    Note: alpha + beta + gamma should ≈ 1.0 for best results
    """
    # Validate weights
    total_weight = alpha + beta + gamma
    if abs(total_weight - 1.0) > 0.01:
        print(f"Warning: Mixing weights sum to {total_weight:.3f}, not 1.0")
        print(f"  alpha={alpha}, beta={beta}, gamma={gamma}")

    # Linear combination
    combined = alpha * emb_suppress + beta * emb_boost + gamma * emb_bg

    # L2-normalize (critical for SSV2A compatibility)
    combined = combined / combined.norm(p=2)

    return combined  # [768]


# ============================================================================
# Audio Evaluation Metrics (FD, FAD, KL)
# Based on AudioEditor paper: https://arxiv.org/abs/2410.02964
# ============================================================================

def calculate_frechet_distance(audio1, audio2, model_name='vggish', sr=16000, device='cuda'):
    """
    Calculate Frechet Distance (FD) between two audio samples or directories.

    FD measures the distance between two multivariate Gaussian distributions
    fitted to audio features. Lower FD indicates more similar audio.

    Args:
        audio1: Path to audio file or directory of audio files
        audio2: Path to audio file or directory of audio files
        model_name: Feature extractor to use ('vggish' or 'pann')
        sr: Sample rate (default 16000)
        device: 'cuda' or 'cpu'

    Returns:
        float: Frechet Distance score (lower is better)

    Note:
        - For single files: extracts features and computes FD
        - For directories: computes statistics over all files then FD
    """
    import sys
    import os
    import tempfile
    import shutil

    # Save original sys.argv to prevent argparse conflicts
    original_argv = sys.argv.copy()

    try:
        # Temporarily set minimal argv during import to avoid argparse conflicts
        sys.argv = ['audio_eval']

        try:
            from audioldm_eval import EvaluationHelper
        except ImportError:
            raise ImportError(
                "audioldm_eval is required. Install with: "
                "pip install git+https://github.com/haoheliu/audioldm_eval.git"
            )
    finally:
        # Always restore original argv
        sys.argv = original_argv

    # Determine if inputs are files or directories
    is_file1 = os.path.isfile(audio1)
    is_file2 = os.path.isfile(audio2)

    # Initialize evaluator
    evaluator = EvaluationHelper(sr, device)

    if is_file1 and is_file2:
        # For single files, create temporary directories
        with tempfile.TemporaryDirectory() as tmpdir1, \
             tempfile.TemporaryDirectory() as tmpdir2:

            # Copy files to temp directories with same name for matching
            shutil.copy(audio1, os.path.join(tmpdir1, 'audio.wav'))
            shutil.copy(audio2, os.path.join(tmpdir2, 'audio.wav'))

            # Calculate metrics (returns dict with FD, FAD, KL, IS)
            metrics = evaluator.calculate_metrics(
                tmpdir1, tmpdir2,
                same_name=True
            )
            fd_score = metrics['frechet_distance']
    else:
        # Directory comparison
        metrics = evaluator.calculate_metrics(
            audio1, audio2,
            same_name=False
        )
        fd_score = metrics['frechet_distance']

    return fd_score


def calculate_frechet_audio_distance(audio1, audio2, sr=16000, device='cuda'):
    """
    Calculate Frechet Audio Distance (FAD) between two audio samples or directories.

    FAD uses VGGish features to compute the Frechet distance between audio distributions.
    This is the standard metric used in the AudioEditor paper.

    Args:
        audio1: Path to audio file or directory of audio files
        audio2: Path to audio file or directory of audio files
        sr: Sample rate (default 16000)
        device: 'cuda' or 'cpu'

    Returns:
        float: FAD score (lower is better)

    Reference:
        AudioEditor paper uses VGGish model for FAD computation
    """
    import sys
    import os

    # Save original sys.argv to prevent argparse conflicts
    # Some audio libraries have argparse at module level which conflicts with our script
    original_argv = sys.argv.copy()

    try:
        # Temporarily set minimal argv during import to avoid argparse conflicts
        sys.argv = ['audio_eval']

        try:
            from frechet_audio_distance import FrechetAudioDistance
        except ImportError:
            raise ImportError(
                "frechet_audio_distance is required. Install with: "
                "pip install frechet-audio-distance"
            )
    finally:
        # Always restore original argv
        sys.argv = original_argv

    # Determine if inputs are files or directories
    is_file1 = os.path.isfile(audio1)
    is_file2 = os.path.isfile(audio2)

    # Initialize FAD calculator with VGGish model
    frechet = FrechetAudioDistance(
        model_name="vggish",
        sample_rate=sr,
        use_pca=False,
        use_activation=False,
        verbose=False
    )

    if is_file1 and is_file2:
        # For single files, create temporary directories
        import tempfile
        import shutil

        with tempfile.TemporaryDirectory() as tmpdir1, \
             tempfile.TemporaryDirectory() as tmpdir2:

            # Copy files to temp directories
            shutil.copy(audio1, tmpdir1)
            shutil.copy(audio2, tmpdir2)

            # Calculate FAD on directories
            fad_score = frechet.score(tmpdir1, tmpdir2)
    else:
        # Directory comparison
        fad_score = frechet.score(audio1, audio2)

    return fad_score


def calculate_kl_divergence(audio1, audio2, model_name='pann', sr=16000, device='cuda'):
    """
    Calculate Kullback-Leibler (KL) divergence between two audio samples or directories.

    KL divergence measures how one probability distribution diverges from another.
    Uses PANNs classifier features as specified in AudioEditor paper.

    Args:
        audio1: Path to audio file or directory of audio files
        audio2: Path to audio file or directory of audio files
        model_name: Feature extractor to use (default 'pann')
        sr: Sample rate (default 16000)
        device: 'cuda' or 'cpu'

    Returns:
        float: KL divergence score (lower is better, 0 = identical distributions)

    Reference:
        AudioEditor paper uses PANNs classifier for KL computation
    """
    import sys
    import os
    import tempfile
    import shutil

    # Save original sys.argv to prevent argparse conflicts
    original_argv = sys.argv.copy()

    try:
        # Temporarily set minimal argv during import to avoid argparse conflicts
        sys.argv = ['audio_eval']

        try:
            from audioldm_eval import EvaluationHelper
        except ImportError:
            raise ImportError(
                "audioldm_eval is required. Install with: "
                "pip install git+https://github.com/haoheliu/audioldm_eval.git"
            )
    finally:
        # Always restore original argv
        sys.argv = original_argv

    # Determine if inputs are files or directories
    is_file1 = os.path.isfile(audio1)
    is_file2 = os.path.isfile(audio2)

    # Initialize evaluator
    evaluator = EvaluationHelper(sr, device)

    if is_file1 and is_file2:
        # For single files, create temporary directories
        with tempfile.TemporaryDirectory() as tmpdir1, \
             tempfile.TemporaryDirectory() as tmpdir2:

            # Copy files to temp directories with same name for matching
            shutil.copy(audio1, os.path.join(tmpdir1, 'audio.wav'))
            shutil.copy(audio2, os.path.join(tmpdir2, 'audio.wav'))

            # Calculate metrics (returns dict with FD, FAD, KL, IS)
            metrics = evaluator.calculate_metrics(
                tmpdir1, tmpdir2,
                same_name=True
            )
            kl_score = metrics['kullback_leibler_divergence_softmax']
    else:
        # Directory comparison
        metrics = evaluator.calculate_metrics(
            audio1, audio2,
            same_name=False
        )
        kl_score = metrics['kullback_leibler_divergence_softmax']

    return kl_score
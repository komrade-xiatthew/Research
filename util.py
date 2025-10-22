import torch
import torch.nn.functional as F

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

def downsample_mask_to_patch_weights(mask, emb, target_res=224):
    """
    mask: B x H x W in {0,1} from SAM2 image space
    emb:  B x T x D CLIP tokens with CLS at index 0
    returns: B x (T-1) weights aligned to patch tokens emb[:,1:,:]
    """
    Bm, H, W = mask.shape
    Be, T, _ = emb.shape
    assert Bm == Be, "batch sizes must match"
    n_tokens = T - 1
    g = int(round(n_tokens ** 0.5))       # grid size, e.g. 49 -> 7
    assert g * g == n_tokens, "patch tokens must form a square grid"
    new_size, crop_box = _compute_resize_and_center_crop(H, W, target=target_res)
    mask_224 = _apply_resize_crop_mask(mask, new_size, crop_box)      # B x 224 x 224
    patch = target_res // g
    w = F.avg_pool2d(mask_224.unsqueeze(1), kernel_size=patch, stride=patch)  # B x 1 x g x g
    return w.flatten(1)   # B x (g*g) = B x (T-1)

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
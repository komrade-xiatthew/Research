import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import torchvision
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import util

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    #torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )


np.random.seed(3)
sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)

predictor = SAM2ImagePredictor(sam2_model)


def get_mask(image_path, point_x, point_y, point_labels):
    """
    Generate binary segmentation mask using SAM2 for a single image.

    Args:
        image_path: String path to image file
        point_x: X coordinate for SAM2 point prompt
        point_y: Y coordinate for SAM2 point prompt
        point_labels: List [1] for foreground or [0] for background

    Returns:
        Binary numpy array [H, W] with values {0.0, 1.0}
        representing the segmentation mask
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)

    # Set image in predictor
    predictor.set_image(image_np)

    # Create point prompt
    point_coords = np.array([[point_x, point_y]])
    point_labels_arr = np.array(point_labels)

    # Predict mask (single mask output for simplicity)
    masks, scores, logits = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels_arr,
        multimask_output=False  # Return single best mask
    )

    # Return binary mask (first mask, convert to float)
    return masks[0].astype(np.float32)  # [H, W]


def get_masks(images, x, y, labels):
    """
    Generate masks for multiple images (batch processing).

    Args:
        images: List of image file paths
        x: X coordinate for point prompt
        y: Y coordinate for point prompt
        labels: Point labels for SAM2

    Returns:
        List of binary numpy arrays [H, W]
    """
    res = []
    for image in images:
        img_arr = np.array(Image.open(image).convert("RGB"))
        predictor.set_image(img_arr)

        masks, scores, logits = predictor.predict(
            point_coords=np.array([[x, y]]),
            point_labels=np.array(labels),
            multimask_output=True,
        )
        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]
        logits = logits[sorted_ind]

        res.append(masks[0])
    return res
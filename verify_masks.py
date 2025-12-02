"""
Interactive tool to verify SAM2 mask coordinates before running main.py

Usage:
    python verify_masks.py --image images/original.png --x 500 --y 400
    
Or run interactively:
    python verify_masks.py
"""

import argparse
import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from mask import get_mask


def visualize_point_and_mask(image_path, x, y, labels=[1], save_path=None):
    """
    Generate SAM2 mask and visualize it with the clicked point.
    
    Args:
        image_path: Path to the image
        x: X coordinate of the point prompt
        y: Y coordinate of the point prompt
        labels: Point labels (1 for foreground, 0 for background)
        save_path: Optional path to save the visualization
    """
    # Load image
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)
    print(f"Image size: {img_array.shape[1]} x {img_array.shape[0]} (W x H)")
    print(f"Point: ({x}, {y})")
    
    # Generate mask
    print("Generating SAM2 mask...")
    try:
        mask = get_mask(image_path, x, y, labels)
        print(f"Mask shape: {mask.shape}")
        print(f"Mask coverage: {mask.sum() / mask.size * 100:.1f}% of image")
    except Exception as e:
        print(f"Error generating mask: {e}")
        return None
    
    # Resize mask to match image if needed
    if mask.shape != (img_array.shape[0], img_array.shape[1]):
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        mask_img = mask_img.resize((img_array.shape[1], img_array.shape[0]), Image.NEAREST)
        mask = np.array(mask_img) / 255.0
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image with point marker
    axes[0].imshow(img_array)
    axes[0].scatter([x], [y], c='red', s=200, marker='*', edgecolors='white', linewidths=2)
    axes[0].set_title(f"Original Image\nPoint: ({x}, {y})", fontsize=12)
    axes[0].axis('off')
    
    # Mask only
    axes[1].imshow(mask, cmap='gray')
    axes[1].scatter([x], [y], c='red', s=200, marker='*', edgecolors='white', linewidths=2)
    axes[1].set_title("SAM2 Mask", fontsize=12)
    axes[1].axis('off')
    
    # Overlay
    overlay = img_array.copy().astype(float)
    mask_colored = np.zeros_like(img_array, dtype=float)
    mask_colored[:, :, 0] = 255  # Red
    mask_colored[:, :, 1] = 50   # Some green
    alpha = 0.5
    mask_3d = np.stack([mask, mask, mask], axis=-1)
    overlay = (overlay * (1 - alpha * mask_3d) + mask_colored * alpha * mask_3d).astype(np.uint8)
    
    axes[2].imshow(overlay)
    axes[2].scatter([x], [y], c='yellow', s=200, marker='*', edgecolors='black', linewidths=2)
    axes[2].set_title("Mask Overlay", fontsize=12)
    axes[2].axis('off')
    
    plt.suptitle(f"SAM2 Segmentation: {os.path.basename(image_path)}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved visualization to: {save_path}")
        plt.close()
    else:
        plt.savefig("outputs/verify_mask_temp.png", dpi=150, bbox_inches='tight')
        print(f"✓ Saved visualization to: outputs/verify_mask_temp.png")
        plt.close()
    
    return mask


def interactive_mode():
    """Run in interactive mode to test multiple coordinates."""
    print("=" * 60)
    print("SAM2 Mask Verification Tool")
    print("=" * 60)
    print("\nThis tool helps you find the correct coordinates for SAM2 masks.")
    print("The coordinates should point to the CENTER of the object you want to segment.\n")
    
    # Ensure output directory exists
    os.makedirs("outputs", exist_ok=True)
    
    while True:
        print("\n" + "-" * 40)
        image_path = input("Enter image path (or 'q' to quit): ").strip()
        
        if image_path.lower() == 'q':
            print("Exiting...")
            break
        
        if not os.path.exists(image_path):
            print(f"Error: File not found: {image_path}")
            continue
        
        # Show image size
        img = Image.open(image_path)
        print(f"Image size: {img.size[0]} x {img.size[1]} (W x H)")
        
        try:
            x = int(input("Enter X coordinate: ").strip())
            y = int(input("Enter Y coordinate: ").strip())
        except ValueError:
            print("Error: Please enter valid integer coordinates")
            continue
        
        # Generate and visualize
        save_name = f"verify_{os.path.basename(image_path).split('.')[0]}_{x}_{y}.png"
        save_path = os.path.join("outputs", save_name)
        
        mask = visualize_point_and_mask(image_path, x, y, save_path=save_path)
        
        if mask is not None:
            print(f"\n✓ Check the visualization at: {save_path}")
            print("  - The RED STAR shows your click point")
            print("  - The MASK shows what SAM2 segmented")
            print("  - Adjust coordinates if the mask doesn't match your target object")


def main():
    parser = argparse.ArgumentParser(description="Verify SAM2 mask coordinates")
    parser.add_argument("--image", "-i", type=str, help="Path to image")
    parser.add_argument("--x", type=int, help="X coordinate")
    parser.add_argument("--y", type=int, help="Y coordinate")
    parser.add_argument("--output", "-o", type=str, help="Output path for visualization")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs("outputs", exist_ok=True)
    
    if args.interactive or (args.image is None and args.x is None and args.y is None):
        interactive_mode()
    else:
        if not all([args.image, args.x is not None, args.y is not None]):
            print("Error: Please provide --image, --x, and --y")
            print("Or run without arguments for interactive mode")
            sys.exit(1)
        
        save_path = args.output or f"outputs/verify_{os.path.basename(args.image).split('.')[0]}_{args.x}_{args.y}.png"
        visualize_point_and_mask(args.image, args.x, args.y, save_path=save_path)


if __name__ == "__main__":
    main()
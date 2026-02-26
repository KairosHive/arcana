"""
Test script for LAB color transfer (Reinhard et al. method).

Usage:
    python test_lab_transfer.py --style path/to/style.jpg --content_dir path/to/images/ --output_dir path/to/output/

This applies the color palette from the style image to all images in content_dir.
"""

import numpy as np
import cv2
import os
import argparse
from pathlib import Path
from PIL import Image
import time


def lab_color_transfer(content: np.ndarray, style: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """
    Transfer colors from style image to content using LAB mean/std matching.
    
    Args:
        content: Content image (BGR, uint8)
        style: Style/reference image (BGR, uint8)  
        strength: Blend strength 0.0-1.0 (1.0 = full transfer)
    
    Returns:
        Result image (BGR, uint8)
    """
    # Convert to LAB (float32 for precision)
    content_lab = cv2.cvtColor(content, cv2.COLOR_BGR2LAB).astype(np.float32)
    style_lab = cv2.cvtColor(style, cv2.COLOR_BGR2LAB).astype(np.float32)
    
    # Compute channel-wise mean and std
    content_mean, content_std = cv2.meanStdDev(content_lab)
    style_mean, style_std = cv2.meanStdDev(style_lab)
    
    # Flatten to 1D arrays
    content_mean = content_mean.flatten()
    content_std = content_std.flatten()
    style_mean = style_mean.flatten()
    style_std = style_std.flatten()
    
    # Avoid division by zero
    content_std = np.where(content_std < 1e-6, 1e-6, content_std)
    
    # Vectorized transfer: normalize then rescale
    # x' = (x - content_mean) * (style_std / content_std) + style_mean
    result_lab = content_lab.copy()
    for c in range(3):
        result_lab[:, :, c] = (content_lab[:, :, c] - content_mean[c]) * (style_std[c] / content_std[c]) + style_mean[c]
    
    # Blend with original based on strength
    if strength < 1.0:
        result_lab = content_lab * (1 - strength) + result_lab * strength
    
    # Clip to valid LAB range and convert back
    result_lab[:, :, 0] = np.clip(result_lab[:, :, 0], 0, 255)  # L: 0-255 in OpenCV
    result_lab[:, :, 1] = np.clip(result_lab[:, :, 1], 0, 255)  # A: 0-255 (centered at 128)
    result_lab[:, :, 2] = np.clip(result_lab[:, :, 2], 0, 255)  # B: 0-255 (centered at 128)
    
    result_bgr = cv2.cvtColor(result_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    return result_bgr


def lab_color_transfer_pil(content: Image.Image, style: Image.Image, strength: float = 1.0) -> Image.Image:
    """PIL wrapper for lab_color_transfer."""
    content_bgr = cv2.cvtColor(np.array(content), cv2.COLOR_RGB2BGR)
    style_bgr = cv2.cvtColor(np.array(style), cv2.COLOR_RGB2BGR)
    
    result_bgr = lab_color_transfer(content_bgr, style_bgr, strength)
    
    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(result_rgb)


def batch_transfer(style_path: str, content_dir: str, output_dir: str, 
                   strength: float = 1.0, max_images: int = 20):
    """
    Apply LAB color transfer to a batch of images.
    
    Args:
        style_path: Path to style/reference image
        content_dir: Directory containing content images
        output_dir: Directory to save results
        strength: Transfer strength 0.0-1.0
        max_images: Maximum number of images to process
    """
    # Load style image
    style_img = cv2.imread(style_path)
    if style_img is None:
        raise ValueError(f"Could not load style image: {style_path}")
    
    print(f"Style image: {style_path}")
    print(f"Style size: {style_img.shape[1]}x{style_img.shape[0]}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of image files
    content_path = Path(content_dir)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = [f for f in content_path.iterdir() 
                   if f.suffix.lower() in image_extensions][:max_images]
    
    print(f"\nProcessing {len(image_files)} images from {content_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Strength: {strength}")
    print("-" * 50)
    
    total_time = 0
    
    for i, img_path in enumerate(image_files):
        start = time.time()
        
        # Load content image
        content_img = cv2.imread(str(img_path))
        if content_img is None:
            print(f"  [{i+1}/{len(image_files)}] Skipped (could not load): {img_path.name}")
            continue
        
        # Apply transfer
        result_img = lab_color_transfer(content_img, style_img, strength)
        
        # Save result
        output_path = Path(output_dir) / f"lab_{img_path.stem}{img_path.suffix}"
        cv2.imwrite(str(output_path), result_img)
        
        elapsed = time.time() - start
        total_time += elapsed
        
        print(f"  [{i+1}/{len(image_files)}] {img_path.name} -> {output_path.name} ({elapsed:.3f}s)")
    
    print("-" * 50)
    print(f"Done! Processed {len(image_files)} images in {total_time:.2f}s")
    print(f"Average: {total_time/len(image_files)*1000:.1f}ms per image")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LAB Color Transfer (Reinhard method)")
    parser.add_argument("--style", required=True, help="Path to style/reference image")
    parser.add_argument("--content_dir", required=True, help="Directory of content images")
    parser.add_argument("--output_dir", default="test_output/lab_transfer", help="Output directory")
    parser.add_argument("--strength", type=float, default=1.0, help="Transfer strength 0.0-1.0")
    parser.add_argument("--max", type=int, default=20, help="Maximum images to process")
    
    args = parser.parse_args()
    
    batch_transfer(
        style_path=args.style,
        content_dir=args.content_dir,
        output_dir=args.output_dir,
        strength=args.strength,
        max_images=args.max,
    )

# palette_demo.py — Demo script for palette matching methods
# Creates a visual comparison of the 3 palette matching approaches

import os
import sys
import random
import shutil
from glob import glob
from tqdm import tqdm

from palette import (
    extract_all_palette_features,
    histogram_similarity,
    moments_distance,
    emd_palette_distance,
    render_palette_strip,
    dominant_colors_to_rgb,
)

import numpy as np
import cv2


def get_image_files(folder: str) -> list:
    """Get all image files from folder recursively."""
    extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
    all_files = glob(os.path.join(folder, "**", "*"), recursive=True)
    return [f for f in all_files if os.path.splitext(f.lower())[1] in extensions]


def create_comparison_image(
    source_path: str,
    matches: list,
    source_palette: np.ndarray,
    method_name: str,
    output_path: str,
    thumb_size: int = 200,
):
    """
    Create a single comparison image showing source + top matches with palettes.
    
    Args:
        source_path: Path to source image
        matches: List of (path, score) tuples
        source_palette: Dominant colors array (N, 4)
        method_name: Name of the method for title
        output_path: Where to save the comparison image
        thumb_size: Size for thumbnails
    """
    n_matches = len(matches)
    cols = min(6, n_matches + 1)  # source + matches
    rows = (n_matches + 1 + cols - 1) // cols
    
    # Each cell: thumbnail + palette strip
    cell_h = thumb_size + 60  # thumbnail + palette + label
    cell_w = thumb_size + 20
    
    canvas = np.ones((rows * cell_h + 40, cols * cell_w + 20, 3), dtype=np.uint8) * 40
    
    # Title
    cv2.putText(canvas, f"Method: {method_name}", (10, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    def draw_cell(img_path, palette, label, row, col):
        x = col * cell_w + 10
        y = row * cell_h + 40
        
        # Load and resize image
        img = cv2.imread(img_path)
        if img is None:
            return
        h, w = img.shape[:2]
        scale = thumb_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        thumb = cv2.resize(img, (new_w, new_h))
        
        # Center thumbnail
        x_offset = (thumb_size - new_w) // 2
        y_offset = (thumb_size - new_h) // 2
        canvas[y + y_offset:y + y_offset + new_h, 
               x + x_offset:x + x_offset + new_w] = thumb
        
        # Draw palette strip
        strip = render_palette_strip(palette, width=thumb_size, height=20, n_colors=10)
        strip_bgr = cv2.cvtColor(strip, cv2.COLOR_RGB2BGR)
        canvas[y + thumb_size + 5:y + thumb_size + 25, x:x + thumb_size] = strip_bgr
        
        # Label
        cv2.putText(canvas, label[:20], (x, y + thumb_size + 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    # Draw source
    draw_cell(source_path, source_palette, "SOURCE", 0, 0)
    
    # Draw matches
    for i, (path, score) in enumerate(matches):
        row = (i + 1) // cols
        col = (i + 1) % cols
        
        # Load palette for this match
        feats = extract_all_palette_features(path)
        palette = feats['dominant']
        
        # Format score
        if method_name == "Histogram":
            label = f"#{i+1} sim={score:.3f}"
        else:
            label = f"#{i+1} dist={score:.2f}"
        
        draw_cell(path, palette, label, row, col)
    
    cv2.imwrite(output_path, canvas)


def run_demo(
    images_folder: str,
    output_folder: str,
    n_sources: int = 10,
    n_matches: int = 10,
    max_images: int = 500,
    seed: int = 42,
):
    """
    Run the palette matching demo.
    
    Args:
        images_folder: Folder containing images to analyze
        output_folder: Where to save results
        n_sources: Number of random source images to test
        n_matches: Number of top matches per method
        max_images: Maximum images to load (for speed)
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    # Get all images
    print(f"Scanning {images_folder} for images...")
    all_images = get_image_files(images_folder)
    print(f"Found {len(all_images)} images total")
    
    # Subsample for speed
    if max_images and len(all_images) > max_images:
        all_images = random.sample(all_images, max_images)
        print(f"Subsampled to {max_images} images for speed")
    
    if len(all_images) < n_sources + n_matches:
        print(f"[ERROR] Need at least {n_sources + n_matches} images, found {len(all_images)}")
        return
    
    # Select random source images
    source_images = random.sample(all_images, n_sources)
    
    # Extract features for ALL images (this is the slow part)
    print("\nExtracting palette features for all images...")
    all_features = {}
    for path in tqdm(all_images, desc="Extracting palettes"):
        try:
            all_features[path] = extract_all_palette_features(path)
        except Exception as e:
            print(f"[WARN] Failed on {path}: {e}")
    
    print(f"\nSuccessfully extracted features for {len(all_features)} images")
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Process each source image
    for src_idx, source_path in enumerate(source_images):
        print(f"\n[{src_idx + 1}/{n_sources}] Processing: {os.path.basename(source_path)}")
        
        src_feats = all_features.get(source_path)
        if src_feats is None:
            print(f"  Skipping - no features extracted")
            continue
        
        # Create subfolder
        subfolder = os.path.join(output_folder, f"sample_{src_idx + 1:02d}")
        os.makedirs(subfolder, exist_ok=True)
        
        # Copy source image
        src_ext = os.path.splitext(source_path)[1]
        shutil.copy(source_path, os.path.join(subfolder, f"source{src_ext}"))
        
        # Save source palette strip
        strip = render_palette_strip(src_feats['dominant'], width=600, height=60, n_colors=20)
        cv2.imwrite(os.path.join(subfolder, "source_palette.png"), 
                    cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
        
        # Candidates = all except source
        candidates = {p: f for p, f in all_features.items() if p != source_path}
        
        # --- Method 1: Histogram (cosine similarity - higher is better) ---
        hist_scores = []
        for path, feats in candidates.items():
            sim = histogram_similarity(src_feats['histogram'], feats['histogram'])
            hist_scores.append((path, sim))
        hist_scores.sort(key=lambda x: -x[1])  # Descending (higher = better)
        hist_top = hist_scores[:n_matches]
        
        hist_folder = os.path.join(subfolder, "histogram")
        os.makedirs(hist_folder, exist_ok=True)
        for i, (path, score) in enumerate(hist_top):
            ext = os.path.splitext(path)[1]
            shutil.copy(path, os.path.join(hist_folder, f"{i + 1:02d}_sim{score:.3f}{ext}"))
        
        # Create comparison image
        create_comparison_image(
            source_path, hist_top, src_feats['dominant'],
            "Histogram (Cosine)", os.path.join(subfolder, "histogram_comparison.jpg")
        )
        
        # --- Method 2: Moments (euclidean distance - lower is better) ---
        moments_scores = []
        for path, feats in candidates.items():
            dist = moments_distance(src_feats['moments'], feats['moments'])
            moments_scores.append((path, dist))
        moments_scores.sort(key=lambda x: x[1])  # Ascending (lower = better)
        moments_top = moments_scores[:n_matches]
        
        moments_folder = os.path.join(subfolder, "moments")
        os.makedirs(moments_folder, exist_ok=True)
        for i, (path, score) in enumerate(moments_top):
            ext = os.path.splitext(path)[1]
            shutil.copy(path, os.path.join(moments_folder, f"{i + 1:02d}_dist{score:.1f}{ext}"))
        
        create_comparison_image(
            source_path, moments_top, src_feats['dominant'],
            "Moments (Euclidean)", os.path.join(subfolder, "moments_comparison.jpg")
        )
        
        # --- Method 3: EMD (Earth Mover's Distance - lower is better) ---
        emd_scores = []
        for path, feats in candidates.items():
            dist = emd_palette_distance(src_feats['dominant'], feats['dominant'], n_colors=10)
            emd_scores.append((path, dist))
        emd_scores.sort(key=lambda x: x[1])  # Ascending (lower = better)
        emd_top = emd_scores[:n_matches]
        
        emd_folder = os.path.join(subfolder, "emd")
        os.makedirs(emd_folder, exist_ok=True)
        for i, (path, score) in enumerate(emd_top):
            ext = os.path.splitext(path)[1]
            shutil.copy(path, os.path.join(emd_folder, f"{i + 1:02d}_emd{score:.1f}{ext}"))
        
        create_comparison_image(
            source_path, emd_top, src_feats['dominant'],
            "EMD (10 colors)", os.path.join(subfolder, "emd_comparison.jpg")
        )
        
        print(f"  Saved results to {subfolder}")
    
    print(f"\n{'='*60}")
    print(f"Demo complete! Results saved to: {output_folder}")
    print(f"Each sample folder contains:")
    print(f"  - source.* : The query image")
    print(f"  - source_palette.png : Extracted dominant colors")
    print(f"  - histogram/ : Top {n_matches} by LAB histogram similarity")
    print(f"  - moments/   : Top {n_matches} by color moments distance")
    print(f"  - emd/       : Top {n_matches} by Earth Mover's Distance")
    print(f"  - *_comparison.jpg : Visual comparison for each method")


if __name__ == "__main__":
    # Default paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_images = os.path.abspath(os.path.join(script_dir, "..", "images", "midjourney"))
    default_output = os.path.join(script_dir, "test_output", "palette_demo")
    
    # Parse arguments
    if len(sys.argv) >= 3:
        images_folder = sys.argv[1]
        output_folder = sys.argv[2]
    elif len(sys.argv) == 2:
        images_folder = sys.argv[1]
        output_folder = default_output
    else:
        images_folder = default_images
        output_folder = default_output
    
    print(f"Palette Matching Demo")
    print(f"{'='*60}")
    print(f"Images folder: {images_folder}")
    print(f"Output folder: {output_folder}")
    print(f"{'='*60}")
    
    run_demo(
        images_folder=images_folder,
        output_folder=output_folder,
        n_sources=10,
        n_matches=10,
        seed=42,
    )

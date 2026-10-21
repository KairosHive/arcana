# style_demo.py — Visual demonstration of style matching methods
# Similar to palette_demo.py but for style features (edge, texture, gram)
#
# Usage:
#   python style_demo.py [--images_dir <path>] [--output_dir <path>] [--n_samples 10] [--max_images 500]

import os
import sys
import random
import argparse
from glob import glob

import cv2
import numpy as np
from tqdm import tqdm

# Import style module
from style import (
    extract_edge_histogram,
    extract_texture_lbp,
    extract_gram_features,
    edge_histogram_similarity,
    texture_lbp_similarity,
    gram_similarity,
    HAS_TORCH,
)

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
DEFAULT_IMAGES_DIR = os.path.join(os.path.dirname(__file__), "..", "images", "midjourney")
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "test_output", "style_demo")


def is_image(path: str) -> bool:
    return os.path.splitext(path.lower())[1] in IMAGE_EXTS


def load_images(images_dir: str, max_images: int = None) -> list:
    """Load all image paths from directory."""
    pattern = os.path.join(images_dir, "**", "*")
    all_files = glob(pattern, recursive=True)
    paths = [p for p in all_files if is_image(p)]
    
    if max_images and len(paths) > max_images:
        print(f"Subsampling from {len(paths)} to {max_images} images for speed")
        paths = random.sample(paths, max_images)
    
    return paths


def extract_all_styles(paths: list, include_gram: bool = True) -> dict:
    """Extract style features for all images."""
    features = {}
    
    for path in tqdm(paths, desc="Extracting styles"):
        try:
            edge = extract_edge_histogram(path)
            lbp = extract_texture_lbp(path)
            gram = None
            if include_gram and HAS_TORCH:
                try:
                    gram = extract_gram_features(path)
                except Exception as e:
                    pass  # Skip gram if it fails
            
            features[path] = {
                'edge_histogram': edge,
                'texture_lbp': lbp,
                'gram': gram,
            }
        except Exception as e:
            print(f"[WARN] Failed to extract style for {os.path.basename(path)}: {e}")
    
    return features


def rank_by_method(ref_path: str, ref_features: dict, all_features: dict, method: str, top_n: int = 10) -> list:
    """Rank all images by similarity to reference using specified method."""
    ref_feat = ref_features.get(method)
    if ref_feat is None:
        return []
    
    scores = []
    for path, feats in all_features.items():
        if path == ref_path:
            continue
        feat = feats.get(method)
        if feat is None:
            continue
        
        if method == 'edge_histogram':
            sim = edge_histogram_similarity(ref_feat, feat)
        elif method == 'texture_lbp':
            sim = texture_lbp_similarity(ref_feat, feat)
        elif method == 'gram':
            sim = gram_similarity(ref_feat, feat)
        else:
            continue
        
        scores.append((path, sim))
    
    scores.sort(key=lambda x: -x[1])
    return scores[:top_n]


def create_thumbnail(image_path: str, size: int = 150) -> np.ndarray:
    """Create a square thumbnail."""
    img = cv2.imread(image_path)
    if img is None:
        return np.zeros((size, size, 3), dtype=np.uint8)
    
    h, w = img.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Pad to square
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    y_off = (size - new_h) // 2
    x_off = (size - new_w) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = img
    
    return canvas


def render_style_visualization(ref_path: str, ref_features: dict) -> np.ndarray:
    """Create a visualization of the style features for an image."""
    img = cv2.imread(ref_path)
    if img is None:
        return np.zeros((200, 400, 3), dtype=np.uint8)
    
    # Resize reference
    h, w = img.shape[:2]
    scale = 200 / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    img_small = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Create canvas
    canvas = np.zeros((200, 600, 3), dtype=np.uint8)
    
    # Place image
    y_off = (200 - new_h) // 2
    canvas[y_off:y_off+new_h, 0:new_w] = img_small
    
    # Draw edge histogram as bars (simplified)
    edge_hist = ref_features.get('edge_histogram')
    if edge_hist is not None:
        # Take first 16 bins (one cell) as example
        bins = edge_hist[:16]
        max_val = bins.max() + 1e-8
        for i, v in enumerate(bins):
            bar_h = int((v / max_val) * 80)
            x = new_w + 20 + i * 10
            cv2.rectangle(canvas, (x, 100-bar_h), (x+8, 100), (100, 200, 100), -1)
        cv2.putText(canvas, "Edge Hist", (new_w + 20, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    # Draw LBP histogram as bars
    lbp_hist = ref_features.get('texture_lbp')
    if lbp_hist is not None:
        # Take first 10 bins as example
        bins = lbp_hist[:10]
        max_val = bins.max() + 1e-8
        for i, v in enumerate(bins):
            bar_h = int((v / max_val) * 80)
            x = new_w + 200 + i * 12
            cv2.rectangle(canvas, (x, 100-bar_h), (x+10, 100), (200, 150, 100), -1)
        cv2.putText(canvas, "LBP Texture", (new_w + 200, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    # Gram indicator
    gram = ref_features.get('gram')
    if gram is not None:
        cv2.putText(canvas, f"Gram: {len(gram)} dims", (new_w + 380, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 255), 1)
    
    return canvas


def create_comparison_grid(ref_path: str, ranked: list, method_name: str, 
                           thumb_size: int = 120, cols: int = 5) -> np.ndarray:
    """Create a grid showing reference + top matches."""
    n_results = len(ranked)
    rows = (n_results + cols - 1) // cols + 1  # +1 for header row
    
    width = cols * (thumb_size + 10) + 10
    height = rows * (thumb_size + 30) + 60
    
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Header
    cv2.putText(canvas, f"Style: {method_name}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Reference image (larger)
    ref_thumb = create_thumbnail(ref_path, thumb_size + 20)
    canvas[50:50+thumb_size+20, 10:10+thumb_size+20] = ref_thumb
    cv2.putText(canvas, "REF", (10, 50+thumb_size+35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    # Results grid
    start_y = 50 + thumb_size + 50
    for i, (path, score) in enumerate(ranked):
        row = i // cols
        col = i % cols
        x = 10 + col * (thumb_size + 10)
        y = start_y + row * (thumb_size + 30)
        
        if y + thumb_size + 20 > height:
            break
        
        thumb = create_thumbnail(path, thumb_size)
        canvas[y:y+thumb_size, x:x+thumb_size] = thumb
        
        # Score label
        label = f"{score:.3f}"
        cv2.putText(canvas, label, (x, y + thumb_size + 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    
    return canvas


def run_demo(
    images_dir: str = DEFAULT_IMAGES_DIR,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    n_samples: int = 10,
    max_images: int = 500,
    include_gram: bool = True,
):
    """Run the style matching demo."""
    print(f"\n{'='*60}")
    print("STYLE MATCHING DEMO")
    print(f"{'='*60}")
    print(f"Images dir: {images_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Samples: {n_samples}")
    print(f"Max images: {max_images}")
    print(f"Include Gram: {include_gram and HAS_TORCH}")
    print(f"PyTorch available: {HAS_TORCH}")
    print(f"{'='*60}\n")
    
    # Load image paths
    paths = load_images(images_dir, max_images=max_images)
    if len(paths) < 2:
        print("[ERROR] Need at least 2 images")
        return
    
    print(f"Loaded {len(paths)} images")
    
    # Extract style features
    features = extract_all_styles(paths, include_gram=include_gram)
    valid_paths = [p for p in paths if p in features]
    print(f"Successfully extracted features from {len(valid_paths)} images")
    
    # Pick random samples
    n_samples = min(n_samples, len(valid_paths))
    sample_paths = random.sample(valid_paths, n_samples)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each sample
    methods = ['edge_histogram', 'texture_lbp']
    if include_gram and HAS_TORCH:
        methods.append('gram')
    
    for i, ref_path in enumerate(sample_paths):
        print(f"\n[{i+1}/{n_samples}] Processing: {os.path.basename(ref_path)}")
        
        ref_features = features[ref_path]
        sample_dir = os.path.join(output_dir, f"sample_{i+1:02d}")
        os.makedirs(sample_dir, exist_ok=True)
        
        # Copy reference image
        ref_img = cv2.imread(ref_path)
        if ref_img is not None:
            cv2.imwrite(os.path.join(sample_dir, f"source{os.path.splitext(ref_path)[1]}"), ref_img)
        
        # Style visualization
        style_viz = render_style_visualization(ref_path, ref_features)
        cv2.imwrite(os.path.join(sample_dir, "source_style.png"), style_viz)
        
        # Create comparison for each method
        for method in methods:
            method_dir = os.path.join(sample_dir, method)
            os.makedirs(method_dir, exist_ok=True)
            
            # Rank by this method
            ranked = rank_by_method(ref_path, ref_features, features, method, top_n=20)
            
            if not ranked:
                print(f"  [{method}] No results")
                continue
            
            print(f"  [{method}] Top match: {ranked[0][1]:.4f}")
            
            # Save top matches
            for j, (match_path, score) in enumerate(ranked[:10]):
                match_img = cv2.imread(match_path)
                if match_img is not None:
                    ext = os.path.splitext(match_path)[1]
                    cv2.imwrite(os.path.join(method_dir, f"{j+1:02d}_{score:.3f}{ext}"), match_img)
            
            # Create grid comparison
            grid = create_comparison_grid(ref_path, ranked[:15], method)
            cv2.imwrite(os.path.join(sample_dir, f"{method}_comparison.jpg"), grid)
    
    print(f"\n{'='*60}")
    print(f"Demo complete! Results saved to: {output_dir}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Style Matching Demo")
    parser.add_argument("--images_dir", type=str, default=DEFAULT_IMAGES_DIR,
                        help="Directory containing images")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Output directory for results")
    parser.add_argument("--n_samples", type=int, default=10,
                        help="Number of random samples to process")
    parser.add_argument("--max_images", type=int, default=500,
                        help="Maximum images to load (for speed)")
    parser.add_argument("--no_gram", action="store_true",
                        help="Skip Gram matrix features (faster)")
    args = parser.parse_args()
    
    run_demo(
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        n_samples=args.n_samples,
        max_images=args.max_images,
        include_gram=not args.no_gram,
    )


if __name__ == "__main__":
    main()

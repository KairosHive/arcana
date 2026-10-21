"""
Test script for datamoshing techniques.

Runs each of the 3 methods on 10 image pairs:
1. Pixel Displacement
2. Motion Vector Injection  
3. FFT Phase Swap

Total: 30 pairs processed
"""

import os
import time
from pathlib import Path
from PIL import Image
import random

from datamosh import (
    pixel_displacement,
    motion_vector_injection,
    fft_phase_swap,
    fft_phase_swap_advanced
)


def load_and_resize(path: str, max_size: int = 768) -> Image.Image:
    """Load image and resize to manageable size."""
    img = Image.open(path).convert('RGB')
    
    # Resize maintaining aspect ratio
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
    
    return img


def get_image_pairs(image_folder: str, n_pairs: int = 10, seed: int = 42) -> list:
    """Get random pairs of images from folder."""
    random.seed(seed)
    
    images = sorted([
        f for f in os.listdir(image_folder)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    
    if len(images) < 2:
        raise ValueError(f"Need at least 2 images, found {len(images)}")
    
    pairs = []
    for _ in range(n_pairs):
        # Pick two different images
        img1, img2 = random.sample(images, 2)
        pairs.append((
            os.path.join(image_folder, img1),
            os.path.join(image_folder, img2)
        ))
    
    return pairs


def test_pixel_displacement(pairs: list, output_dir: str):
    """Test pixel displacement on image pairs."""
    print("\n" + "="*60)
    print("Testing PIXEL DISPLACEMENT")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    times = []
    
    for i, (path1, path2) in enumerate(pairs):
        name1 = Path(path1).stem
        name2 = Path(path2).stem
        
        print(f"  Pair {i+1}: {name1} + {name2}", end=" ")
        
        img1 = load_and_resize(path1)
        img2 = load_and_resize(path2)
        
        start = time.time()
        
        # Test with different strengths
        result_normal = pixel_displacement(img1, img2, strength=1.0)
        result_strong = pixel_displacement(img1, img2, strength=2.5)
        result_noisy = pixel_displacement(img1, img2, strength=1.5, noise_amount=0.3)
        
        elapsed = time.time() - start
        times.append(elapsed)
        
        # Save results
        result_normal.save(os.path.join(output_dir, f"{i+1:02d}_{name1}_{name2}_normal.jpg"), quality=95)
        result_strong.save(os.path.join(output_dir, f"{i+1:02d}_{name1}_{name2}_strong.jpg"), quality=95)
        result_noisy.save(os.path.join(output_dir, f"{i+1:02d}_{name1}_{name2}_noisy.jpg"), quality=95)
        
        print(f"[{elapsed:.2f}s]")
    
    avg_time = sum(times) / len(times)
    total_time = sum(times)
    print(f"\n  Total: {total_time:.2f}s | Avg per pair: {avg_time:.2f}s")
    
    return total_time


def test_motion_vector(pairs: list, output_dir: str):
    """Test motion vector injection on image pairs."""
    print("\n" + "="*60)
    print("Testing MOTION VECTOR INJECTION")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    times = []
    
    for i, (path1, path2) in enumerate(pairs):
        name1 = Path(path1).stem
        name2 = Path(path2).stem
        
        print(f"  Pair {i+1}: {name1} + {name2}", end=" ")
        
        img1 = load_and_resize(path1)
        img2 = load_and_resize(path2)
        
        start = time.time()
        
        # Test with different parameters
        result_subtle = motion_vector_injection(img1, img2, exaggeration=2.0, trail_steps=3)
        result_trails = motion_vector_injection(img1, img2, exaggeration=4.0, trail_steps=7, trail_decay=0.6)
        result_extreme = motion_vector_injection(img1, img2, exaggeration=8.0, trail_steps=10, trail_decay=0.8)
        
        elapsed = time.time() - start
        times.append(elapsed)
        
        # Save results
        result_subtle.save(os.path.join(output_dir, f"{i+1:02d}_{name1}_{name2}_subtle.jpg"), quality=95)
        result_trails.save(os.path.join(output_dir, f"{i+1:02d}_{name1}_{name2}_trails.jpg"), quality=95)
        result_extreme.save(os.path.join(output_dir, f"{i+1:02d}_{name1}_{name2}_extreme.jpg"), quality=95)
        
        print(f"[{elapsed:.2f}s]")
    
    avg_time = sum(times) / len(times)
    total_time = sum(times)
    print(f"\n  Total: {total_time:.2f}s | Avg per pair: {avg_time:.2f}s")
    
    return total_time


def test_fft_phase_swap(pairs: list, output_dir: str):
    """Test FFT phase swap on image pairs."""
    print("\n" + "="*60)
    print("Testing FFT PHASE SWAP")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    times = []
    
    for i, (path1, path2) in enumerate(pairs):
        name1 = Path(path1).stem
        name2 = Path(path2).stem
        
        print(f"  Pair {i+1}: {name1} + {name2}", end=" ")
        
        img1 = load_and_resize(path1)
        img2 = load_and_resize(path2)
        
        start = time.time()
        
        # Test basic phase swap
        result_full = fft_phase_swap(img1, img2, phase_blend=1.0)
        result_partial = fft_phase_swap(img1, img2, phase_blend=0.5)
        
        # Test advanced frequency-selective swap
        result_advanced = fft_phase_swap_advanced(
            img1, img2, 
            low_freq_phase_blend=0.2,
            high_freq_phase_blend=0.9,
            cutoff_ratio=0.15
        )
        
        elapsed = time.time() - start
        times.append(elapsed)
        
        # Save results
        result_full.save(os.path.join(output_dir, f"{i+1:02d}_{name1}_{name2}_full.jpg"), quality=95)
        result_partial.save(os.path.join(output_dir, f"{i+1:02d}_{name1}_{name2}_partial.jpg"), quality=95)
        result_advanced.save(os.path.join(output_dir, f"{i+1:02d}_{name1}_{name2}_advanced.jpg"), quality=95)
        
        print(f"[{elapsed:.2f}s]")
    
    avg_time = sum(times) / len(times)
    total_time = sum(times)
    print(f"\n  Total: {total_time:.2f}s | Avg per pair: {avg_time:.2f}s")
    
    return total_time


def main():
    print("="*60)
    print("DATAMOSH TEST SUITE")
    print("="*60)
    
    # Setup paths
    script_dir = Path(__file__).parent
    image_folder = script_dir.parent / "images" / "japan"
    output_base = script_dir / "test_output" / "datamosh"
    
    print(f"Image source: {image_folder}")
    print(f"Output: {output_base}")
    
    # Generate 10 pairs for each method (using different seeds for variety)
    pairs_pixel = get_image_pairs(str(image_folder), n_pairs=10, seed=42)
    pairs_motion = get_image_pairs(str(image_folder), n_pairs=10, seed=123)
    pairs_fft = get_image_pairs(str(image_folder), n_pairs=10, seed=456)
    
    print(f"\nGenerated 30 total pairs (10 per method)")
    
    # Run tests
    total_start = time.time()
    
    t1 = test_pixel_displacement(pairs_pixel, str(output_base / "pixel_displacement"))
    t2 = test_motion_vector(pairs_motion, str(output_base / "motion_vector"))
    t3 = test_fft_phase_swap(pairs_fft, str(output_base / "fft_phase_swap"))
    
    total_elapsed = time.time() - total_start
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"  Pixel Displacement:    {t1:.2f}s (10 pairs)")
    print(f"  Motion Vector:         {t2:.2f}s (10 pairs)")
    print(f"  FFT Phase Swap:        {t3:.2f}s (10 pairs)")
    print(f"  ----------------------------------------")
    print(f"  TOTAL:                 {total_elapsed:.2f}s (30 pairs)")
    print(f"  Average per pair:      {total_elapsed/30:.2f}s")
    print(f"\n  Results saved to: {output_base}")


if __name__ == "__main__":
    main()

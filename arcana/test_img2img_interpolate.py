"""
Test SD-Turbo img2img interpolation between image pairs.

Uses the same model as inject_poetry but for image-to-image morphing.
Approach: Start from image A, use img2img with increasing strength toward B.
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
import random
import os
import time
from diffusers import StableDiffusionImg2ImgPipeline

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

IMAGE_DIR = Path("images/japan")
OUTPUT_DIR = Path("arcana/test_output/img2img_interpolate")
N_PAIRS = 5  # Fewer pairs since this is slower
N_STEPS = 5  # Number of interpolation steps between images
MAX_SIZE = 512
STRENGTH = 0.6  # How much diffusion changes each step (0.4-0.7 works well)

# ─────────────────────────────────────────────────────────────────────────────
# Load Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def load_pipeline():
    """Load SD-Turbo img2img pipeline (same as inject_poetry)."""
    print("Loading SD-Turbo pipeline...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "stabilityai/sd-turbo",
        torch_dtype=torch.float16,
        variant="fp16"
    ).to(device)
    
    # Same optimizations as inject_poetry
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except:
        pass
    pipe.enable_vae_slicing()
    pipe.safety_checker = None
    
    print(f"Pipeline loaded on {device}")
    return pipe, device


# ─────────────────────────────────────────────────────────────────────────────
# Image Processing
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_image(img: Image.Image, max_size: int = 512) -> Image.Image:
    """Resize and ensure dimensions are multiples of 8."""
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        w, h = int(w * scale), int(h * scale)
    
    # Make divisible by 8
    w = (w // 8) * 8
    h = (h // 8) * 8
    
    return img.resize((w, h), Image.LANCZOS)


def blend_images(img1: Image.Image, img2: Image.Image, t: float) -> Image.Image:
    """Simple pixel blend between two images."""
    arr1 = np.array(img1).astype(float)
    arr2 = np.array(img2).astype(float)
    blended = (1 - t) * arr1 + t * arr2
    return Image.fromarray(blended.astype(np.uint8))


# ─────────────────────────────────────────────────────────────────────────────
# Interpolation Methods
# ─────────────────────────────────────────────────────────────────────────────

def interpolate_img2img_blend(pipe, img1: Image.Image, img2: Image.Image, 
                               n_steps: int, strength: float = 0.5) -> list[Image.Image]:
    """
    Interpolate using img2img with blended init images.
    
    For each step:
    - Blend img1 and img2 at ratio t
    - Run img2img on the blend to "clean it up"
    
    This creates smoother transitions than pure blending.
    """
    # Ensure same size
    img1 = preprocess_image(img1)
    img2 = preprocess_image(img2)
    
    w1, h1 = img1.size
    w2, h2 = img2.size
    target_w = min(w1, w2)
    target_h = min(h1, h2)
    target_w = (target_w // 8) * 8
    target_h = (target_h // 8) * 8
    
    img1 = img1.resize((target_w, target_h), Image.LANCZOS)
    img2 = img2.resize((target_w, target_h), Image.LANCZOS)
    
    t_values = np.linspace(0, 1, n_steps + 2)
    results = []
    
    for i, t in enumerate(t_values):
        if t == 0:
            results.append(img1)
        elif t == 1:
            results.append(img2)
        else:
            # Blend the two images
            blended = blend_images(img1, img2, t)
            
            # Use img2img to "clean up" the blend
            # Lower strength = preserve more of blend, higher = more creative
            gen = torch.manual_seed(42 + i)
            
            out = pipe(
                prompt="",  # No prompt, just denoise
                image=blended,
                strength=strength,
                num_inference_steps=4,  # SD-Turbo only needs 1-4 steps
                guidance_scale=0.0,  # No guidance for pure denoising
                generator=gen,
            ).images[0]
            
            results.append(out)
    
    return results


def interpolate_img2img_chain(pipe, img1: Image.Image, img2: Image.Image,
                               n_steps: int, strength: float = 0.4) -> list[Image.Image]:
    """
    Interpolate using chained img2img transformations.
    
    Start from img1, progressively transform toward img2 by using
    blended images as targets.
    """
    # Ensure same size
    img1 = preprocess_image(img1)
    img2 = preprocess_image(img2)
    
    w1, h1 = img1.size
    w2, h2 = img2.size
    target_w = min(w1, w2)
    target_h = min(h1, h2)
    target_w = (target_w // 8) * 8
    target_h = (target_h // 8) * 8
    
    img1 = img1.resize((target_w, target_h), Image.LANCZOS)
    img2 = img2.resize((target_w, target_h), Image.LANCZOS)
    
    results = [img1]
    current = img1
    
    for i in range(1, n_steps + 1):
        t = i / (n_steps + 1)
        
        # Target is a blend toward img2
        target_blend = blend_images(img1, img2, t)
        
        # Run img2img from current toward the target
        gen = torch.manual_seed(42 + i)
        
        out = pipe(
            prompt="",
            image=target_blend,
            strength=strength,
            num_inference_steps=4,
            guidance_scale=0.0,
            generator=gen,
        ).images[0]
        
        results.append(out)
        current = out
    
    results.append(img2)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Grid Creation
# ─────────────────────────────────────────────────────────────────────────────

def create_comparison_grid(images: list[Image.Image]) -> Image.Image:
    """Create a horizontal grid of interpolation results."""
    target_h = 256
    resized = []
    for img in images:
        w, h = img.size
        new_w = int(w * target_h / h)
        resized.append(img.resize((new_w, target_h), Image.LANCZOS))
    
    total_w = sum(img.size[0] for img in resized)
    grid = Image.new('RGB', (total_w, target_h))
    
    x = 0
    for img in resized:
        grid.paste(img, (x, 0))
        x += img.size[0]
    
    return grid


def get_random_pairs(image_dir: Path, n_pairs: int) -> list[tuple[Path, Path]]:
    """Get random pairs of images from a directory."""
    extensions = {'.jpg', '.jpeg', '.png', '.webp'}
    images = [f for f in image_dir.iterdir() if f.suffix.lower() in extensions]
    
    pairs = []
    for _ in range(n_pairs):
        pair = random.sample(images, 2)
        pairs.append((pair[0], pair[1]))
    
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    pipe, device = load_pipeline()
    
    print(f"\nSelecting {N_PAIRS} random pairs from {IMAGE_DIR}")
    pairs = get_random_pairs(IMAGE_DIR, N_PAIRS)
    
    print(f"Generating {N_STEPS} interpolation steps per pair")
    print(f"Method: img2img blend with strength={STRENGTH}")
    print(f"Output: {OUTPUT_DIR}\n")
    print("-" * 60)
    
    total_time = 0
    for i, (path1, path2) in enumerate(pairs):
        print(f"\nPair {i+1}/{N_PAIRS}:")
        print(f"  A: {path1.name}")
        print(f"  B: {path2.name}")
        
        img1 = Image.open(path1).convert("RGB")
        img2 = Image.open(path2).convert("RGB")
        
        start = time.time()
        results = interpolate_img2img_blend(pipe, img1, img2, N_STEPS, strength=STRENGTH)
        elapsed = time.time() - start
        total_time += elapsed
        
        print(f"  Generated {len(results)} images in {elapsed:.2f}s")
        
        # Save grid
        grid = create_comparison_grid(results)
        grid_path = OUTPUT_DIR / f"pair_{i+1:02d}_grid.jpg"
        grid.save(grid_path, quality=90)
        
        # Save individual frames
        pair_dir = OUTPUT_DIR / f"pair_{i+1:02d}"
        pair_dir.mkdir(exist_ok=True)
        for j, img in enumerate(results):
            img.save(pair_dir / f"frame_{j:02d}.jpg", quality=90)
        
        print(f"  Saved: {grid_path}")
    
    print("\n" + "-" * 60)
    print(f"Done! {N_PAIRS} pairs processed in {total_time:.1f}s")
    print(f"Average: {total_time/N_PAIRS:.2f}s per pair ({total_time/(N_PAIRS*(N_STEPS+2)):.2f}s per frame)")
    print(f"Output: {OUTPUT_DIR.absolute()}")


if __name__ == "__main__":
    main()

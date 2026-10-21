"""
Test VAE latent interpolation between image pairs.

Uses SD-Turbo's VAE (same as inject_poetry) to encode images to latents,
interpolate in latent space (SLERP), and decode back to images.
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
import random
import os
import time

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

IMAGE_DIR = Path("images/japan")  # Change to your image folder
OUTPUT_DIR = Path("arcana/test_output/vae_interpolate_japan")
N_PAIRS = 10
N_STEPS = 5  # Number of interpolation steps between images (not counting endpoints)
MAX_SIZE = 512  # Resize images to this max dimension for speed

# ─────────────────────────────────────────────────────────────────────────────
# SLERP (Spherical Linear Interpolation)
# ─────────────────────────────────────────────────────────────────────────────

def slerp(v0: torch.Tensor, v1: torch.Tensor, t: float) -> torch.Tensor:
    """
    Spherical linear interpolation between two tensors.
    Better than linear interpolation for high-dimensional latents.
    """
    # Flatten for dot product
    v0_flat = v0.flatten()
    v1_flat = v1.flatten()
    
    # Normalize
    v0_norm = v0_flat / torch.norm(v0_flat)
    v1_norm = v1_flat / torch.norm(v1_flat)
    
    # Angle between vectors
    dot = torch.clamp(torch.dot(v0_norm, v1_norm), -1.0, 1.0)
    theta = torch.acos(dot)
    
    # If vectors are very close, fall back to linear
    if theta < 1e-5:
        return (1 - t) * v0 + t * v1
    
    # SLERP
    sin_theta = torch.sin(theta)
    s0 = torch.sin((1 - t) * theta) / sin_theta
    s1 = torch.sin(t * theta) / sin_theta
    
    result = s0 * v0_flat + s1 * v1_flat
    return result.reshape(v0.shape)


def lerp(v0: torch.Tensor, v1: torch.Tensor, t: float) -> torch.Tensor:
    """Simple linear interpolation."""
    return (1 - t) * v0 + t * v1


# ─────────────────────────────────────────────────────────────────────────────
# VAE Interpolation
# ─────────────────────────────────────────────────────────────────────────────

def load_vae():
    """Load SD-Turbo's VAE."""
    from diffusers import AutoencoderKL
    
    print("Loading SD-Turbo VAE...")
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-turbo",
        subfolder="vae",
        torch_dtype=torch.float16
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = vae.to(device)
    vae.eval()
    print(f"VAE loaded on {device}")
    return vae, device


def preprocess_image(img: Image.Image, max_size: int = 512) -> Image.Image:
    """Resize and ensure dimensions are multiples of 8 (VAE requirement)."""
    # Resize if needed
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        w, h = int(w * scale), int(h * scale)
    
    # Make divisible by 8
    w = (w // 8) * 8
    h = (h // 8) * 8
    
    return img.resize((w, h), Image.LANCZOS)


def image_to_latent(vae, img: Image.Image, device: str) -> torch.Tensor:
    """Encode an image to VAE latent space."""
    # Preprocess
    img = preprocess_image(img)
    
    # To tensor: [0, 255] -> [-1, 1]
    img_tensor = torch.from_numpy(np.array(img)).float() / 127.5 - 1.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    img_tensor = img_tensor.to(device, dtype=torch.float16)
    
    # Encode
    with torch.no_grad():
        latent = vae.encode(img_tensor).latent_dist.sample()
        latent = latent * vae.config.scaling_factor
    
    return latent


def latent_to_image(vae, latent: torch.Tensor) -> Image.Image:
    """Decode a VAE latent back to an image."""
    with torch.no_grad():
        latent = latent / vae.config.scaling_factor
        decoded = vae.decode(latent).sample
    
    # To image: [-1, 1] -> [0, 255]
    decoded = decoded.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
    decoded = ((decoded + 1) * 127.5).clip(0, 255).astype(np.uint8)
    
    return Image.fromarray(decoded)


def interpolate_images(vae, img1: Image.Image, img2: Image.Image, 
                       n_steps: int, device: str, use_slerp: bool = True) -> list[Image.Image]:
    """
    Generate interpolated images between two source images.
    
    Args:
        vae: VAE model
        img1, img2: Source images
        n_steps: Number of intermediate steps (not counting endpoints)
        device: torch device
        use_slerp: Use SLERP (True) or LERP (False)
    
    Returns:
        List of images: [img1, interp1, interp2, ..., img2]
    """
    # Preprocess to same size (use smaller of the two)
    img1_proc = preprocess_image(img1)
    img2_proc = preprocess_image(img2)
    
    # Ensure same dimensions by resizing to common size
    w1, h1 = img1_proc.size
    w2, h2 = img2_proc.size
    target_w = min(w1, w2)
    target_h = min(h1, h2)
    # Make divisible by 8
    target_w = (target_w // 8) * 8
    target_h = (target_h // 8) * 8
    
    img1_proc = img1_proc.resize((target_w, target_h), Image.LANCZOS)
    img2_proc = img2_proc.resize((target_w, target_h), Image.LANCZOS)
    
    # Encode both images
    latent1 = image_to_latent(vae, img1_proc, device)
    latent2 = image_to_latent(vae, img2_proc, device)
    
    # Generate interpolation steps
    interp_fn = slerp if use_slerp else lerp
    t_values = np.linspace(0, 1, n_steps + 2)  # Include 0.0 and 1.0
    
    results = []
    for t in t_values:
        if t == 0:
            results.append(img1_proc)  # Use processed (same-size) version
        elif t == 1:
            results.append(img2_proc)  # Use processed (same-size) version
        else:
            interp_latent = interp_fn(latent1, latent2, t)
            results.append(latent_to_image(vae, interp_latent))
    
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main Test
# ─────────────────────────────────────────────────────────────────────────────

def get_random_pairs(image_dir: Path, n_pairs: int) -> list[tuple[Path, Path]]:
    """Get random pairs of images from a directory."""
    extensions = {'.jpg', '.jpeg', '.png', '.webp'}
    images = [f for f in image_dir.iterdir() if f.suffix.lower() in extensions]
    
    if len(images) < n_pairs * 2:
        print(f"Warning: Only {len(images)} images found, using with replacement")
    
    pairs = []
    for _ in range(n_pairs):
        pair = random.sample(images, 2)
        pairs.append((pair[0], pair[1]))
    
    return pairs


def create_comparison_grid(images: list[Image.Image], pair_idx: int) -> Image.Image:
    """Create a horizontal grid of interpolation results."""
    # Resize all to same size
    target_h = 256
    resized = []
    for img in images:
        w, h = img.size
        new_w = int(w * target_h / h)
        resized.append(img.resize((new_w, target_h), Image.LANCZOS))
    
    # Create grid
    total_w = sum(img.size[0] for img in resized)
    grid = Image.new('RGB', (total_w, target_h))
    
    x = 0
    for img in resized:
        grid.paste(img, (x, 0))
        x += img.size[0]
    
    return grid


def main():
    # Setup
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load VAE
    vae, device = load_vae()
    
    # Get random pairs
    print(f"\nSelecting {N_PAIRS} random pairs from {IMAGE_DIR}")
    pairs = get_random_pairs(IMAGE_DIR, N_PAIRS)
    
    print(f"Generating {N_STEPS} interpolation steps per pair")
    print(f"Output: {OUTPUT_DIR}\n")
    print("-" * 60)
    
    total_time = 0
    for i, (path1, path2) in enumerate(pairs):
        print(f"\nPair {i+1}/{N_PAIRS}:")
        print(f"  A: {path1.name}")
        print(f"  B: {path2.name}")
        
        # Load images
        img1 = Image.open(path1).convert("RGB")
        img2 = Image.open(path2).convert("RGB")
        
        # Interpolate
        start = time.time()
        results = interpolate_images(vae, img1, img2, N_STEPS, device, use_slerp=True)
        elapsed = time.time() - start
        total_time += elapsed
        
        print(f"  Generated {len(results)} images in {elapsed:.2f}s")
        
        # Save grid
        grid = create_comparison_grid(results, i)
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

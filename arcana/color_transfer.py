"""
Color Transfer using ModFlows (Modulated Normalizing Flows).

This module provides a clean API for transferring colors from a style/reference
image to a content image using the ModFlows neural ODE approach.

Based on: "Color Transfer with Modulated Flows" (AAAI 2025)
Paper: https://arxiv.org/abs/2503.19062

Usage:
    from arcana.color_transfer import transfer_colors, get_device_info
    
    # Check available device
    info = get_device_info()
    print(info)  # {'device': 'cuda', 'gpu_name': 'NVIDIA GeForce RTX 4090', ...}
    
    # Transfer colors
    result = transfer_colors(
        content="path/to/image.jpg",  # or PIL Image
        style="path/to/reference.jpg",
        strength=0.8,
        max_size=1024
    )
    result.save("output.jpg")
"""

import os
import sys
import subprocess
from typing import Union, Optional, Tuple
from pathlib import Path

import numpy as np
from PIL import Image

# Lazy imports for torch (heavy)
_torch = None
_encoder = None
_device = None

# Path to modflows directory (sibling to arcana/)
MODFLOWS_DIR = Path(__file__).parent.parent / "modflows"
CHECKPOINT_DIR = MODFLOWS_DIR / "modflows_color_encoder"
CHECKPOINT_NAMES = [
    "modflows_color_encoder_B6_dim_8195_iter_700000.pt",
    "modflows_color_encoder_B6_dim_8195_iter_751001.pt",
]


def get_device_info() -> dict:
    """
    Get information about available compute devices.
    
    Returns:
        dict with keys:
            - device: 'cuda' or 'cpu'
            - cuda_available: bool
            - gpu_name: str or None
            - gpu_memory_gb: float or None
            - torch_version: str
            - cuda_version: str or None
    """
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    
    info = {
        "device": "cuda" if _torch.cuda.is_available() else "cpu",
        "cuda_available": _torch.cuda.is_available(),
        "gpu_name": None,
        "gpu_memory_gb": None,
        "torch_version": _torch.__version__,
        "cuda_version": None,
    }
    
    if _torch.cuda.is_available():
        info["gpu_name"] = _torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = _torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if hasattr(_torch.version, 'cuda'):
            info["cuda_version"] = _torch.version.cuda
    
    return info


def check_cuda_installation() -> dict:
    """
    Check CUDA installation and provide installation guidance if needed.
    
    Returns:
        dict with:
            - is_cuda: bool - whether current torch has CUDA
            - recommendation: str - what to do if CUDA not available
            - install_command: str or None - pip command to install CUDA torch
    """
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    
    result = {
        "is_cuda": _torch.cuda.is_available(),
        "recommendation": None,
        "install_command": None,
    }
    
    if result["is_cuda"]:
        result["recommendation"] = "CUDA is available and working."
        return result
    
    # Check if NVIDIA GPU exists
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        if output.returncode == 0 and output.stdout.strip():
            gpu_name = output.stdout.strip().split('\n')[0]
            result["recommendation"] = (
                f"GPU detected ({gpu_name}) but PyTorch is CPU-only. "
                "Install CUDA-enabled PyTorch for ~10-20x speedup."
            )
            # Detect CUDA version from nvidia-smi
            result["install_command"] = (
                "pip uninstall torch torchvision -y && "
                "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124"
            )
        else:
            result["recommendation"] = "No NVIDIA GPU detected. Using CPU (slower but works)."
    except Exception:
        result["recommendation"] = "Could not detect GPU. Using CPU."
    
    return result


def _ensure_modflows_available():
    """Ensure modflows source is in path."""
    src_path = str(MODFLOWS_DIR / "src")
    if src_path not in sys.path:
        sys.path.insert(0, str(MODFLOWS_DIR))
    
    # Check if source exists
    if not (MODFLOWS_DIR / "src" / "encoder.py").exists():
        raise ImportError(
            f"ModFlows source not found at {MODFLOWS_DIR}/src/. "
            "Please ensure the modflows directory exists with src/ subdirectory."
        )


def _find_checkpoint() -> Path:
    """Find the checkpoint file, downloading if necessary."""
    # Check existing checkpoints
    for name in CHECKPOINT_NAMES:
        path = CHECKPOINT_DIR / name
        if path.exists():
            return path
    
    # Try to download from HuggingFace
    if not CHECKPOINT_DIR.exists():
        print("ModFlows checkpoint not found. Downloading from HuggingFace...")
        try:
            subprocess.run(["git", "lfs", "install"], cwd=str(MODFLOWS_DIR), check=True)
            subprocess.run(
                ["git", "clone", "https://huggingface.co/MariaLarchenko/modflows_color_encoder"],
                cwd=str(MODFLOWS_DIR), check=True
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Failed to download checkpoint: {e}\n"
                "Please download manually:\n"
                "  cd modflows\n"
                "  git lfs install\n"
                "  git clone https://huggingface.co/MariaLarchenko/modflows_color_encoder"
            )
    
    # Check again after download
    for name in CHECKPOINT_NAMES:
        path = CHECKPOINT_DIR / name
        if path.exists():
            return path
    
    raise FileNotFoundError(
        f"No checkpoint found in {CHECKPOINT_DIR}. Expected one of: {CHECKPOINT_NAMES}"
    )


def _get_encoder():
    """Get or create the cached encoder instance."""
    global _encoder, _device, _torch
    
    if _encoder is not None:
        return _encoder, _device
    
    _ensure_modflows_available()
    
    if _torch is None:
        import torch
        _torch = torch
    
    from src.encoder import Encoder
    
    # Determine device
    _device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
    
    # Load encoder
    checkpoint_path = _find_checkpoint()
    _encoder = Encoder(k_dim=8195, input_dim=4, hidden=1024, output_dim=3, device=_device)
    _encoder.load_state_dict(_torch.load(str(checkpoint_path), map_location=_device, weights_only=True))
    _encoder.eval()
    
    return _encoder, _device


def transfer_colors(
    content: Union[str, Path, Image.Image],
    style: Union[str, Path, Image.Image],
    strength: float = 1.0,
    steps: int = 8,
    max_size: int = 1024,
    full_res_output: bool = False,
) -> Image.Image:
    """
    Transfer colors from style image to content image.
    
    Args:
        content: Content image path or PIL Image (structure to preserve)
        style: Style/reference image path or PIL Image (colors to transfer)
        strength: Transfer strength 0.0-1.0 (default: 1.0)
        steps: Number of flow steps 2-100 (default: 8, more steps = smoother)
        max_size: Maximum dimension for processing (default: 1024)
        full_res_output: If True, use LUT to output at content's original resolution
    
    Returns:
        PIL.Image.Image with transferred colors
    """
    _ensure_modflows_available()
    from src.inference import run_inference
    
    encoder, device = _get_encoder()
    
    # Handle PIL Image inputs by saving to temp file (modflows expects paths)
    import tempfile
    temp_files = []
    
    def ensure_path(img, prefix):
        if isinstance(img, Image.Image):
            fd, path = tempfile.mkstemp(suffix=".jpg", prefix=prefix)
            os.close(fd)
            img.save(path, quality=95)
            temp_files.append(path)
            return path
        return str(img)
    
    content_path = ensure_path(content, "content_")
    style_path = ensure_path(style, "style_")
    
    try:
        # Get original content dimensions
        content_img = Image.open(content_path)
        orig_w, orig_h = content_img.size
        content_img.close()
        
        # Calculate compression factor
        compress = None
        if max_size > 0 and max(orig_w, orig_h) > max_size:
            compress = max(orig_w, orig_h) / max_size
        
        # Run the flow
        _, _, styled, _ = run_inference(
            encoder, device, content_path, style_path,
            compress=compress, enc_steps=steps, strength=strength
        )
        
        # Optionally upscale to full resolution using 1D LUT
        if full_res_output and compress is not None and compress > 1:
            styled = _apply_lut_fullres(content_path, styled, orig_w, orig_h)
        
        return styled
        
    finally:
        # Cleanup temp files
        for path in temp_files:
            try:
                os.unlink(path)
            except:
                pass


def _apply_lut_fullres(content_path: str, styled_low: Image.Image, 
                       orig_w: int, orig_h: int) -> Image.Image:
    """
    Apply color transfer at full resolution using 1D LUT built from low-res result.
    
    This is much faster than processing the full image through the neural ODE.
    """
    # Build per-channel 1D LUTs from the low-res transformation
    content_low = Image.open(content_path)
    content_low = content_low.resize(styled_low.size, Image.LANCZOS)
    
    orig_arr = np.array(content_low)
    styled_arr = np.array(styled_low)
    
    lut = np.zeros((3, 256), dtype=np.float32)
    count = np.zeros((3, 256), dtype=np.float32)
    
    for c in range(3):
        np.add.at(lut[c], orig_arr[..., c].ravel(), styled_arr[..., c].ravel())
        np.add.at(count[c], orig_arr[..., c].ravel(), 1)
    
    # Average where we have samples, identity otherwise
    for c in range(3):
        mask = count[c] > 0
        lut[c, mask] /= count[c, mask]
        lut[c, ~mask] = np.arange(256)[~mask]
    
    lut = np.clip(lut, 0, 255).astype(np.uint8)
    
    # Apply LUT to full resolution content
    content_full = np.array(Image.open(content_path))
    
    result = np.stack([
        lut[0][content_full[..., 0]],
        lut[1][content_full[..., 1]],
        lut[2][content_full[..., 2]],
    ], axis=2)
    
    return Image.fromarray(result)


def batch_transfer(
    content: Union[str, Path, Image.Image],
    styles: list[Union[str, Path, Image.Image]],
    strength: float = 1.0,
    steps: int = 8,
    max_size: int = 1024,
) -> list[Image.Image]:
    """
    Transfer colors from multiple style images to a single content image.
    
    More efficient than calling transfer_colors() multiple times as the
    encoder is only loaded once.
    
    Args:
        content: Content image (structure to preserve)
        styles: List of style images (colors to transfer)
        strength: Transfer strength 0.0-1.0
        steps: Number of flow steps
        max_size: Maximum dimension for processing
    
    Returns:
        List of PIL Images with transferred colors
    """
    results = []
    for style in styles:
        result = transfer_colors(content, style, strength, steps, max_size)
        results.append(result)
    return results


# Module-level availability check
COLOR_TRANSFER_AVAILABLE = False
COLOR_TRANSFER_ERROR = None

try:
    _ensure_modflows_available()
    COLOR_TRANSFER_AVAILABLE = True
except ImportError as e:
    COLOR_TRANSFER_ERROR = str(e)

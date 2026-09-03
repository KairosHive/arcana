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
import importlib
import subprocess
from typing import Union, Optional, Tuple
from pathlib import Path

import numpy as np
from PIL import Image

# Lazy imports for torch (heavy)
_torch = None
_encoder = None
_device = None

# ── where ModFlows lives ─────────────────────────────────────────────────────
# Two different things with two different lifetimes, and a frozen build puts
# them in two different places:
#
#   the source (src/encoder.py, ~120 KB)  read-only, ships inside the app
#   the checkpoint (~229 MB .pt)          too big to ship, downloaded once into
#                                         the user's writable data directory
#
# In a dev checkout both sit under <repo>/modflows, which is what the original
# single MODFLOWS_DIR assumed. That assumption breaks when frozen: __file__ is
# then inside the PyInstaller bundle, which is read-only and wiped on
# reinstall, so a checkpoint downloaded there would be lost and a checkpoint
# shipped there would put 229 MB in every download.
try:
    from . import paths as _paths
except ImportError:                                          # loose script
    import paths as _paths

try:
    from . import gpu as _gpu
except ImportError:
    import gpu as _gpu

ENV_MODFLOWS_DIR = "ARCANA_MODFLOWS_DIR"

CHECKPOINT_NAMES = [
    "modflows_color_encoder_B6_dim_8195_iter_700000.pt",
    "modflows_color_encoder_B6_dim_8195_iter_751001.pt",
]
CHECKPOINT_URL = ("https://huggingface.co/MariaLarchenko/modflows_color_encoder"
                  "/resolve/main/{name}?download=true")


def _candidate_dirs() -> list[Path]:
    """Every place ModFlows might be, most specific first."""
    out = []
    env = os.environ.get(ENV_MODFLOWS_DIR)
    if env:
        out.append(Path(os.path.expanduser(env)))
    # Dev checkout: modflows/ beside the arcana package.
    out.append(Path(__file__).parent.parent / "modflows")
    if getattr(sys, "frozen", False):
        # Frozen build. PyInstaller 6 unpacks bundled data into an _internal/
        # directory rather than beside the executable, and sys._MEIPASS is the
        # only reliable way to name it (it is also the temp extraction dir for
        # a onefile build). Checking the executable's own directory as well
        # costs nothing and lets a user drop a modflows/ folder next to the app
        # by hand.
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            out.append(Path(meipass) / "modflows")
        out.append(Path(sys.executable).parent / "modflows")
    # ...and anything downloaded, in the writable data directory.
    out.append(Path(_paths.data_dir()) / "modflows")
    seen, uniq = set(), []
    for p in out:
        s = str(p)
        if s not in seen:
            seen.add(s)
            uniq.append(p)
    return uniq


def modflows_source_dir() -> Path | None:
    """The directory holding src/encoder.py, or None if it is not installed."""
    for d in _candidate_dirs():
        if (d / "src" / "encoder.py").exists():
            return d
    return None


def checkpoint_dir() -> Path:
    """Where a checkpoint is, or where one should be downloaded to."""
    for d in _candidate_dirs():
        cd = d / "modflows_color_encoder"
        if any((cd / n).exists() for n in CHECKPOINT_NAMES):
            return cd
    # Nothing found: downloads go to the writable data directory, never into
    # the bundle.
    return Path(_paths.data_dir()) / "modflows" / "modflows_color_encoder"


def checkpoint_path() -> Path | None:
    cd = checkpoint_dir()
    for n in CHECKPOINT_NAMES:
        if (cd / n).exists():
            return cd / n
    return None


def status() -> dict:
    """What the UI needs to explain the state of ModFlows in one line."""
    src = modflows_source_dir()
    ckpt = checkpoint_path()
    return {
        "source": str(src) if src else None,
        "checkpoint": str(ckpt) if ckpt else None,
        "ready": bool(src and ckpt),
        "download_to": str(checkpoint_dir()),
        "download_mb": 229,
    }


def download_checkpoint(progress=None) -> Path:
    """
    Fetch the ModFlows checkpoint over HTTPS.

    This used to shell out to `git lfs install` and `git clone`, which assumes
    both git and git-lfs are on PATH -- true on a developer's machine and false
    on essentially every machine that would run a packaged build. It also
    cloned the whole repository, which carries two 229 MB checkpoints plus LFS
    history: 958 MB on disk for one file that is actually loaded.

    `progress` is called as progress(fraction, message).
    """
    import urllib.request

    name = CHECKPOINT_NAMES[0]
    dest_dir = checkpoint_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / name
    # Download beside the target and rename, so an interrupted download can
    # never be mistaken for a usable checkpoint on the next run.
    tmp = dest.with_suffix(".part")

    req = urllib.request.Request(CHECKPOINT_URL.format(name=name),
                                 headers={"User-Agent": "arcana"})
    with urllib.request.urlopen(req) as resp, open(tmp, "wb") as fh:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
            done += len(chunk)
            if progress and total:
                progress(done / total,
                         f"Downloading colour model ({done >> 20} of {total >> 20} MB)")
    os.replace(tmp, dest)
    return dest


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
        "device": _gpu.device(),
        "cuda_available": _gpu.available(),
        "gpu_name": None,
        "gpu_memory_gb": None,
        "torch_version": _torch.__version__,
        "cuda_version": None,
    }
    
    if _gpu.available():
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
        "is_cuda": _gpu.available(),
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
    """Make `import src.encoder` work, or explain why it cannot."""
    root = modflows_source_dir()
    if root is None:
        looked = "\n  ".join(str(d) for d in _candidate_dirs())
        raise ImportError(
            "ModFlows source not found. Looked in:\n  " + looked +
            "\nUse the LAB (Reinhard) method, which needs no extra download."
        )

    root = str(root)
    if root not in sys.path:
        # The guard used to test for <modflows>/src while inserting <modflows>,
        # so it never matched and sys.path grew on every failed attempt.
        sys.path.insert(0, root)
        # Python caches directory listings per sys.path entry. If this directory
        # did not exist when it was first probed -- the usual case, because the
        # user adds modflows/ while the app is already running -- the cached
        # miss survives and `import src` keeps failing even though the files are
        # now there. Drop the cache so the fresh contents are seen.
        importlib.invalidate_caches()


def _find_checkpoint(progress=None) -> Path:
    """Find the checkpoint, downloading it once if it is not here yet."""
    found = checkpoint_path()
    if found is not None:
        return found

    dest = checkpoint_dir()
    try:
        return download_checkpoint(progress=progress)
    except Exception as e:
        raise FileNotFoundError(
            f"The ModFlows colour model is not installed and could not be "
            f"downloaded ({type(e).__name__}: {e}).\n"
            f"Put {CHECKPOINT_NAMES[0]} in {dest}, or use the LAB (Reinhard) "
            f"method, which needs no download."
        ) from e


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
    _device = _torch.device(_gpu.device())
    
    # Load encoder. Build it in a local first: assigning the module-level
    # _encoder before load_state_dict lets a second concurrent request see a
    # non-None encoder whose weights are still random, and silently colour an
    # image with an untrained network.
    checkpoint_path = _find_checkpoint()
    enc = Encoder(k_dim=8195, input_dim=4, hidden=1024, output_dim=3, device=_device)
    enc.load_state_dict(_torch.load(str(checkpoint_path), map_location=_device, weights_only=True))
    enc.eval()
    _encoder = enc

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

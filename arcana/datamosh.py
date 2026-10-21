"""
Datamoshing effects module.

Implements creative image manipulation techniques:
1. Pixel Displacement - Motion-based warping using optical flow
2. Motion Vector Injection - Exaggerated optical flow for surreal trails
3. FFT Phase Swap - Frequency domain phase exchange for dreamlike hybrids
"""

import numpy as np
import cv2
from PIL import Image
from typing import Tuple, Optional, Union


def _pil_to_cv2(img: Image.Image) -> np.ndarray:
    """Convert PIL Image to OpenCV BGR format."""
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def _cv2_to_pil(img: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR to PIL Image."""
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def _ensure_same_size(img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Resize images to match dimensions (uses smaller of the two)."""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    if (h1, w1) != (h2, w2):
        h = min(h1, h2)
        w = min(w1, w2)
        img1 = cv2.resize(img1, (w, h))
        img2 = cv2.resize(img2, (w, h))
    
    return img1, img2


def compute_optical_flow(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """
    Compute dense optical flow between two images using Farneback method.
    
    Args:
        img1: First image (BGR)
        img2: Second image (BGR)
    
    Returns:
        Flow field of shape (H, W, 2) containing (dx, dy) displacements
    """
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2,
        None,
        pyr_scale=0.5,
        levels=5,
        winsize=15,
        iterations=3,
        poly_n=7,
        poly_sigma=1.5,
        flags=0
    )
    
    return flow


def pixel_displacement(
    source: Union[Image.Image, np.ndarray],
    target: Union[Image.Image, np.ndarray],
    strength: float = 1.0,
    noise_amount: float = 0.0
) -> Image.Image:
    """
    Displace pixels from source image using optical flow derived from target.
    
    Creates glitch effects by warping the source image based on the motion
    patterns detected between source and target.
    
    Args:
        source: Source image (pixels to warp)
        target: Target image (defines motion field)
        strength: Flow multiplier (1.0 = natural, >1 = exaggerated)
        noise_amount: Random noise to add to displacement (0-1 range)
    
    Returns:
        PIL Image with displaced pixels
    """
    # Convert to CV2 if needed
    if isinstance(source, Image.Image):
        source = _pil_to_cv2(source)
    if isinstance(target, Image.Image):
        target = _pil_to_cv2(target)
    
    source, target = _ensure_same_size(source, target)
    h, w = source.shape[:2]
    
    # Compute optical flow
    flow = compute_optical_flow(source, target)
    
    # Apply strength multiplier
    flow = flow * strength
    
    # Add random noise if requested
    if noise_amount > 0:
        noise = np.random.randn(h, w, 2).astype(np.float32) * noise_amount * 20
        flow = flow + noise
    
    # Create coordinate grids for remapping
    x_coords = np.arange(w)
    y_coords = np.arange(h)
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)
    
    # Apply displacement
    map_x = (x_grid + flow[:, :, 0]).astype(np.float32)
    map_y = (y_grid + flow[:, :, 1]).astype(np.float32)
    
    # Remap pixels
    result = cv2.remap(source, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    
    return _cv2_to_pil(result)


def motion_vector_injection(
    source: Union[Image.Image, np.ndarray],
    target: Union[Image.Image, np.ndarray],
    exaggeration: float = 3.0,
    trail_steps: int = 5,
    trail_decay: float = 0.7,
    blend_original: float = 0.3
) -> Image.Image:
    """
    Create surreal motion trails by exaggerating optical flow vectors.
    
    Calculates motion between images, then renders multiple displaced
    copies with decreasing opacity to create trailing ghost effects.
    
    Args:
        source: Source image
        target: Target image (defines motion direction)
        exaggeration: Flow vector multiplier for surreal effect
        trail_steps: Number of trail copies to render
        trail_decay: Opacity decay per step (0-1)
        blend_original: How much of original to keep (0-1)
    
    Returns:
        PIL Image with motion trails
    """
    # Convert to CV2 if needed
    if isinstance(source, Image.Image):
        source = _pil_to_cv2(source)
    if isinstance(target, Image.Image):
        target = _pil_to_cv2(target)
    
    source, target = _ensure_same_size(source, target)
    h, w = source.shape[:2]
    
    # Compute optical flow
    flow = compute_optical_flow(source, target)
    
    # Create coordinate grids
    x_coords = np.arange(w)
    y_coords = np.arange(h)
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)
    
    # Start with weighted original
    result = source.astype(np.float32) * blend_original
    total_weight = blend_original
    
    # Render trail steps with increasing displacement
    for i in range(1, trail_steps + 1):
        # Progressive displacement
        displacement = flow * exaggeration * (i / trail_steps)
        
        map_x = (x_grid + displacement[:, :, 0]).astype(np.float32)
        map_y = (y_grid + displacement[:, :, 1]).astype(np.float32)
        
        warped = cv2.remap(source, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        # Decay weight for trail effect
        weight = trail_decay ** i
        result = result + warped.astype(np.float32) * weight
        total_weight += weight
    
    # Normalize
    result = (result / total_weight).astype(np.uint8)
    
    return _cv2_to_pil(result)


def fft_phase_swap(
    source: Union[Image.Image, np.ndarray],
    target: Union[Image.Image, np.ndarray],
    phase_blend: float = 1.0,
    magnitude_blend: float = 0.0
) -> Image.Image:
    """
    Swap FFT phases between images for dreamlike hybrid effects.
    
    Takes the magnitude spectrum from one image and phase spectrum from
    another, creating surreal combinations where structure comes from
    phase and color/texture from magnitude.
    
    Args:
        source: Source image (provides base magnitude)
        target: Target image (provides phase information)
        phase_blend: How much target phase to use (0=source, 1=target)
        magnitude_blend: How much target magnitude to use (0=source, 1=target)
    
    Returns:
        PIL Image with swapped frequency components
    """
    # Convert to CV2 if needed
    if isinstance(source, Image.Image):
        source = _pil_to_cv2(source)
    if isinstance(target, Image.Image):
        target = _pil_to_cv2(target)
    
    source, target = _ensure_same_size(source, target)
    
    # Process each channel separately
    result_channels = []
    
    for c in range(3):
        src_channel = source[:, :, c].astype(np.float32)
        tgt_channel = target[:, :, c].astype(np.float32)
        
        # Compute FFT
        src_fft = np.fft.fft2(src_channel)
        tgt_fft = np.fft.fft2(tgt_channel)
        
        # Extract magnitude and phase
        src_mag = np.abs(src_fft)
        src_phase = np.angle(src_fft)
        tgt_mag = np.abs(tgt_fft)
        tgt_phase = np.angle(tgt_fft)
        
        # Blend magnitude and phase
        blended_mag = src_mag * (1 - magnitude_blend) + tgt_mag * magnitude_blend
        blended_phase = src_phase * (1 - phase_blend) + tgt_phase * phase_blend
        
        # Reconstruct FFT from blended components
        combined_fft = blended_mag * np.exp(1j * blended_phase)
        
        # Inverse FFT
        reconstructed = np.fft.ifft2(combined_fft)
        reconstructed = np.real(reconstructed)
        
        # Clip to valid range
        reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
        result_channels.append(reconstructed)
    
    # Merge channels
    result = cv2.merge(result_channels)
    
    return _cv2_to_pil(result)


def fft_phase_swap_advanced(
    source: Union[Image.Image, np.ndarray],
    target: Union[Image.Image, np.ndarray],
    low_freq_phase_blend: float = 0.3,
    high_freq_phase_blend: float = 1.0,
    cutoff_ratio: float = 0.1
) -> Image.Image:
    """
    Advanced phase swap with frequency-dependent blending.
    
    Allows different blend ratios for low and high frequencies,
    preserving overall structure while transferring fine details.
    
    Args:
        source: Source image
        target: Target image
        low_freq_phase_blend: Phase blend for low frequencies (0-1)
        high_freq_phase_blend: Phase blend for high frequencies (0-1)
        cutoff_ratio: Frequency cutoff as ratio of image size (0-0.5)
    
    Returns:
        PIL Image with frequency-selective phase swap
    """
    # Convert to CV2 if needed
    if isinstance(source, Image.Image):
        source = _pil_to_cv2(source)
    if isinstance(target, Image.Image):
        target = _pil_to_cv2(target)
    
    source, target = _ensure_same_size(source, target)
    h, w = source.shape[:2]
    
    # Create frequency mask (low freq = center of FFT)
    y_center, x_center = h // 2, w // 2
    y_coords = np.arange(h) - y_center
    x_coords = np.arange(w) - x_center
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)
    
    # Distance from center (normalized)
    dist = np.sqrt(x_grid**2 + y_grid**2)
    max_dist = np.sqrt(x_center**2 + y_center**2)
    dist_normalized = dist / max_dist
    
    # Create smooth blend mask
    cutoff = cutoff_ratio
    blend_mask = np.clip((dist_normalized - cutoff) / (0.5 - cutoff), 0, 1)
    blend_mask = blend_mask * (high_freq_phase_blend - low_freq_phase_blend) + low_freq_phase_blend
    
    # Process each channel
    result_channels = []
    
    for c in range(3):
        src_channel = source[:, :, c].astype(np.float32)
        tgt_channel = target[:, :, c].astype(np.float32)
        
        # Compute centered FFT
        src_fft = np.fft.fftshift(np.fft.fft2(src_channel))
        tgt_fft = np.fft.fftshift(np.fft.fft2(tgt_channel))
        
        # Extract magnitude and phase
        src_mag = np.abs(src_fft)
        src_phase = np.angle(src_fft)
        tgt_phase = np.angle(tgt_fft)
        
        # Frequency-dependent phase blend
        blended_phase = src_phase * (1 - blend_mask) + tgt_phase * blend_mask
        
        # Reconstruct
        combined_fft = src_mag * np.exp(1j * blended_phase)
        
        # Inverse FFT
        reconstructed = np.fft.ifft2(np.fft.ifftshift(combined_fft))
        reconstructed = np.real(reconstructed)
        reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
        result_channels.append(reconstructed)
    
    result = cv2.merge(result_channels)
    
    return _cv2_to_pil(result)


# Convenience dictionary for accessing methods
DATAMOSH_METHODS = {
    'pixel_displacement': pixel_displacement,
    'motion_vector': motion_vector_injection,
    'fft_phase_swap': fft_phase_swap,
    'fft_phase_swap_advanced': fft_phase_swap_advanced,
}


def datamosh(
    source: Union[Image.Image, np.ndarray],
    target: Union[Image.Image, np.ndarray],
    method: str = 'pixel_displacement',
    **kwargs
) -> Image.Image:
    """
    Apply datamoshing effect using specified method.
    
    Args:
        source: Source image
        target: Target image
        method: One of 'pixel_displacement', 'motion_vector', 'fft_phase_swap', 'fft_phase_swap_advanced'
        **kwargs: Method-specific parameters
    
    Returns:
        PIL Image with datamosh effect applied
    """
    if method not in DATAMOSH_METHODS:
        raise ValueError(f"Unknown method: {method}. Available: {list(DATAMOSH_METHODS.keys())}")
    
    return DATAMOSH_METHODS[method](source, target, **kwargs)

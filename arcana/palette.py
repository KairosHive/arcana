# palette.py — Color palette extraction and similarity for Arcana
# Supports:
#   - LAB histogram (512-dim vector, cosine similarity)
#   - Dominant colors via K-means (20 colors + proportions, EMD comparison)
#   - Color moments (9-dim: mean, std, skew per LAB channel)
#
# All methods use CIELAB color space for perceptual uniformity.

import numpy as np
import cv2
from sklearn.cluster import KMeans
from typing import Tuple, Optional
from functools import lru_cache

# Optional: Python Optimal Transport for EMD
try:
    import ot
    HAS_OT = True
except ImportError:
    HAS_OT = False
    ot = None

# Scipy fallback for simpler Wasserstein
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import cdist


# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------
DEFAULT_HISTOGRAM_BINS = 16  # 16^3 = 4096 dimensions (high granularity for storage)
DEFAULT_N_DOMINANT = 32      # Extract top 32 colors (can use fewer at query time)
DEFAULT_RESIZE_DIM = 128     # Downscale for speed
KMEANS_MAX_PIXELS = 10000   # Max pixels for K-means (subsample if larger)


# --------------------------------------------------------------------------------------
# Image Preprocessing
# --------------------------------------------------------------------------------------
def _load_and_prepare(image_or_path, resize_dim: int = DEFAULT_RESIZE_DIM) -> np.ndarray:
    """
    Load image and convert to LAB color space.
    
    Args:
        image_or_path: BGR image (np.ndarray) or path to image file
        resize_dim: Target dimension for resizing (preserves aspect ratio)
    
    Returns:
        LAB image as np.ndarray (H, W, 3), dtype float32, L in [0,100], A/B in [-128,127]
    """
    if isinstance(image_or_path, str):
        img = cv2.imread(image_or_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_or_path}")
    else:
        img = image_or_path
    
    # Resize immediately to reduce memory
    h, w = img.shape[:2]
    scale = resize_dim / max(h, w)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Convert BGR -> LAB
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # Convert to float: L [0-100], A [-128,127], B [-128,127]
    img_lab = img_lab.astype(np.float32)
    img_lab[:, :, 0] = img_lab[:, :, 0] * (100.0 / 255.0)  # L: 0-255 -> 0-100
    img_lab[:, :, 1] = img_lab[:, :, 1] - 128.0             # A: 0-255 -> -128 to 127
    img_lab[:, :, 2] = img_lab[:, :, 2] - 128.0             # B: 0-255 -> -128 to 127
    
    return img_lab


def _get_pixels(img_lab: np.ndarray, max_pixels: int = KMEANS_MAX_PIXELS) -> np.ndarray:
    """
    Flatten image to (N, 3) pixel array, subsampling if needed.
    """
    pixels = img_lab.reshape(-1, 3)
    if len(pixels) > max_pixels:
        indices = np.random.choice(len(pixels), max_pixels, replace=False)
        pixels = pixels[indices]
    return pixels


# --------------------------------------------------------------------------------------
# Method 1: LAB Histogram
# --------------------------------------------------------------------------------------
def extract_lab_histogram(
    image_or_path,
    bins: int = DEFAULT_HISTOGRAM_BINS,
    resize_dim: int = DEFAULT_RESIZE_DIM
) -> np.ndarray:
    """
    Extract a 3D LAB histogram as a flattened vector.
    
    Args:
        image_or_path: BGR image or path
        bins: Number of bins per channel (total dims = bins^3)
        resize_dim: Downscale dimension
    
    Returns:
        Normalized histogram vector of shape (bins^3,), dtype float32
    """
    img_lab = _load_and_prepare(image_or_path, resize_dim)
    
    # Convert back to 0-255 range for cv2.calcHist
    img_hist = img_lab.copy()
    img_hist[:, :, 0] = img_hist[:, :, 0] * (255.0 / 100.0)
    img_hist[:, :, 1] = img_hist[:, :, 1] + 128.0
    img_hist[:, :, 2] = img_hist[:, :, 2] + 128.0
    img_hist = img_hist.astype(np.uint8)
    
    # Compute 3D histogram
    hist = cv2.calcHist(
        [img_hist], [0, 1, 2], None,
        [bins, bins, bins],
        [0, 256, 0, 256, 0, 256]
    )
    
    # Normalize and flatten
    hist = cv2.normalize(hist, hist).flatten().astype(np.float32)
    return hist


def histogram_similarity(hist_a: np.ndarray, hist_b: np.ndarray) -> float:
    """
    Compute cosine similarity between two histogram vectors.
    
    Returns:
        Similarity score in [0, 1], higher = more similar
    """
    norm_a = np.linalg.norm(hist_a)
    norm_b = np.linalg.norm(hist_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(hist_a, hist_b) / (norm_a * norm_b))


# --------------------------------------------------------------------------------------
# Method 2: Dominant Colors (K-means)
# --------------------------------------------------------------------------------------
def extract_dominant_colors(
    image_or_path,
    n_colors: int = DEFAULT_N_DOMINANT,
    resize_dim: int = DEFAULT_RESIZE_DIM,
    max_pixels: int = KMEANS_MAX_PIXELS
) -> np.ndarray:
    """
    Extract dominant colors using K-means clustering in LAB space.
    
    Args:
        image_or_path: BGR image or path
        n_colors: Number of dominant colors to extract
        resize_dim: Downscale dimension
        max_pixels: Max pixels to use for K-means
    
    Returns:
        Array of shape (n_colors, 4) with [L, A, B, proportion] per color,
        sorted by proportion (descending). dtype float32.
    """
    img_lab = _load_and_prepare(image_or_path, resize_dim)
    pixels = _get_pixels(img_lab, max_pixels)
    
    # K-means clustering
    kmeans = KMeans(n_clusters=n_colors, n_init=3, max_iter=100, random_state=42)
    labels = kmeans.fit_predict(pixels)
    centers = kmeans.cluster_centers_  # (n_colors, 3)
    
    # Count pixels per cluster -> proportions
    unique, counts = np.unique(labels, return_counts=True)
    proportions = np.zeros(n_colors, dtype=np.float32)
    for i, count in zip(unique, counts):
        proportions[i] = count
    proportions = proportions / proportions.sum()
    
    # Combine: [L, A, B, proportion]
    palette = np.zeros((n_colors, 4), dtype=np.float32)
    palette[:, :3] = centers
    palette[:, 3] = proportions
    
    # Sort by proportion (descending)
    order = np.argsort(-palette[:, 3])
    palette = palette[order]
    
    return palette


def dominant_colors_to_rgb(palette: np.ndarray) -> np.ndarray:
    """
    Convert dominant colors palette from LAB to RGB for visualization.
    
    Args:
        palette: (N, 4) array [L, A, B, proportion]
    
    Returns:
        (N, 3) array of RGB values in [0, 255], dtype uint8
    """
    n = len(palette)
    lab_colors = palette[:, :3].copy()
    
    # Convert LAB back to 0-255 for OpenCV
    lab_colors[:, 0] = lab_colors[:, 0] * (255.0 / 100.0)
    lab_colors[:, 1] = lab_colors[:, 1] + 128.0
    lab_colors[:, 2] = lab_colors[:, 2] + 128.0
    lab_colors = np.clip(lab_colors, 0, 255).astype(np.uint8)
    
    # Reshape for cv2.cvtColor: (1, N, 3)
    lab_img = lab_colors.reshape(1, n, 3)
    rgb_img = cv2.cvtColor(lab_img, cv2.COLOR_LAB2RGB)
    
    return rgb_img.reshape(n, 3)


def render_palette_strip(
    palette: np.ndarray,
    width: int = 400,
    height: int = 50,
    n_colors: Optional[int] = None
) -> np.ndarray:
    """
    Render a palette as a horizontal color strip image.
    
    Args:
        palette: (N, 4) array [L, A, B, proportion]
        width: Output image width
        height: Output image height
        n_colors: Use top N colors (default: all)
    
    Returns:
        RGB image as np.ndarray (height, width, 3), dtype uint8
    """
    if n_colors is not None:
        palette = palette[:n_colors]
    
    rgb_colors = dominant_colors_to_rgb(palette)
    proportions = palette[:, 3]
    proportions = proportions / proportions.sum()  # Re-normalize after slicing
    
    strip = np.zeros((height, width, 3), dtype=np.uint8)
    x = 0
    for color, prop in zip(rgb_colors, proportions):
        w = max(1, int(prop * width))
        strip[:, x:x+w] = color
        x += w
    
    # Fill remaining pixels with last color
    if x < width:
        strip[:, x:] = rgb_colors[-1]
    
    return strip


# --------------------------------------------------------------------------------------
# Method 3: Color Moments
# --------------------------------------------------------------------------------------
def extract_color_moments(
    image_or_path,
    resize_dim: int = DEFAULT_RESIZE_DIM
) -> np.ndarray:
    """
    Extract color moments (mean, std, skewness) for each LAB channel.
    
    Args:
        image_or_path: BGR image or path
        resize_dim: Downscale dimension
    
    Returns:
        Array of shape (9,): [μL, μA, μB, σL, σA, σB, skewL, skewA, skewB]
    """
    img_lab = _load_and_prepare(image_or_path, resize_dim)
    pixels = img_lab.reshape(-1, 3)
    
    # Mean
    mean = pixels.mean(axis=0)
    
    # Std
    std = pixels.std(axis=0)
    
    # Skewness (third moment)
    skew = np.zeros(3, dtype=np.float32)
    for i in range(3):
        centered = pixels[:, i] - mean[i]
        if std[i] > 1e-8:
            skew[i] = (centered ** 3).mean() / (std[i] ** 3)
    
    moments = np.concatenate([mean, std, skew]).astype(np.float32)
    return moments


def moments_distance(moments_a: np.ndarray, moments_b: np.ndarray) -> float:
    """
    Compute Euclidean distance between color moments.
    
    Returns:
        Distance (lower = more similar)
    """
    return float(np.linalg.norm(moments_a - moments_b))


# --------------------------------------------------------------------------------------
# Method 4: Earth Mover's Distance (EMD)
# --------------------------------------------------------------------------------------
def emd_palette_distance(
    palette_a: np.ndarray,
    palette_b: np.ndarray,
    n_colors: int = 10
) -> float:
    """
    Compute Earth Mover's Distance between two dominant color palettes.
    
    Uses CIELAB Euclidean distance as the ground metric (perceptually uniform).
    Requires the 'pot' (Python Optimal Transport) library for best results.
    
    Args:
        palette_a: (N, 4) array [L, A, B, weight] - reference palette
        palette_b: (N, 4) array [L, A, B, weight] - candidate palette
        n_colors: Use top N colors (5, 10, 15, or 20)
    
    Returns:
        EMD distance (lower = more similar)
    """
    # Take top n_colors
    pa = palette_a[:n_colors].copy()
    pb = palette_b[:n_colors].copy()
    
    # Extract weights and normalize
    wa = pa[:, 3].astype(np.float64)
    wb = pb[:, 3].astype(np.float64)
    wa = wa / wa.sum()
    wb = wb / wb.sum()
    
    # Color positions in LAB space
    xa = pa[:, :3].astype(np.float64)
    xb = pb[:, :3].astype(np.float64)
    
    # Ground distance matrix (CIELAB Euclidean)
    M = cdist(xa, xb, metric='euclidean')
    
    if HAS_OT:
        # Use POT library (more accurate)
        return float(ot.emd2(wa, wb, M))
    else:
        # Fallback: average 1D Wasserstein across channels (less accurate)
        dist = 0.0
        for i in range(3):
            dist += wasserstein_distance(xa[:, i], xb[:, i], wa, wb)
        return dist / 3.0


# --------------------------------------------------------------------------------------
# Combined Palette Extraction
# --------------------------------------------------------------------------------------
def extract_all_palette_features(
    image_or_path,
    histogram_bins: int = DEFAULT_HISTOGRAM_BINS,
    n_dominant: int = DEFAULT_N_DOMINANT,
    resize_dim: int = DEFAULT_RESIZE_DIM
) -> dict:
    """
    Extract all palette features for an image.
    
    Returns:
        Dictionary with keys:
            'histogram': np.ndarray (bins^3,)
            'dominant': np.ndarray (n_dominant, 4) - [L, A, B, proportion]
            'moments': np.ndarray (9,)
    """
    img_lab = _load_and_prepare(image_or_path, resize_dim)
    
    # Histogram
    hist = extract_lab_histogram(img_lab, bins=histogram_bins, resize_dim=resize_dim)
    
    # Dominant colors (need original image, not LAB)
    if isinstance(image_or_path, str):
        dominant = extract_dominant_colors(image_or_path, n_colors=n_dominant, resize_dim=resize_dim)
    else:
        dominant = extract_dominant_colors(image_or_path, n_colors=n_dominant, resize_dim=resize_dim)
    
    # Moments
    moments = extract_color_moments(img_lab, resize_dim=resize_dim)
    
    return {
        'histogram': hist,
        'dominant': dominant,
        'moments': moments,
    }


# --------------------------------------------------------------------------------------
# Batch Processing Utilities
# --------------------------------------------------------------------------------------
def batch_extract_palettes(
    paths: list,
    histogram_bins: int = DEFAULT_HISTOGRAM_BINS,
    n_dominant: int = DEFAULT_N_DOMINANT,
    resize_dim: int = DEFAULT_RESIZE_DIM,
    show_progress: bool = True
) -> dict:
    """
    Extract palette features for a batch of images.
    
    Returns:
        Dictionary mapping path -> palette_dict
    """
    from tqdm import tqdm
    
    results = {}
    iterator = tqdm(paths, desc="Extracting palettes") if show_progress else paths
    
    for path in iterator:
        try:
            results[path] = extract_all_palette_features(
                path,
                histogram_bins=histogram_bins,
                n_dominant=n_dominant,
                resize_dim=resize_dim
            )
        except Exception as e:
            print(f"[WARN] Failed to extract palette for {path}: {e}")
            results[path] = None
    
    return results


# --------------------------------------------------------------------------------------
# Search / Ranking Functions
# --------------------------------------------------------------------------------------
def rank_by_histogram(
    reference_hist: np.ndarray,
    candidate_hists: dict,
    top_n: int = 20
) -> list:
    """
    Rank candidates by histogram cosine similarity.
    
    Args:
        reference_hist: Reference histogram vector
        candidate_hists: Dict mapping path -> histogram vector
        top_n: Number of results to return
    
    Returns:
        List of (path, similarity) tuples, sorted by similarity (descending)
    """
    scores = []
    for path, hist in candidate_hists.items():
        if hist is not None:
            sim = histogram_similarity(reference_hist, hist)
            scores.append((path, sim))
    
    scores.sort(key=lambda x: -x[1])
    return scores[:top_n]


def rank_by_emd(
    reference_palette: np.ndarray,
    candidate_palettes: dict,
    n_colors: int = 10,
    top_n: int = 20
) -> list:
    """
    Rank candidates by EMD (lower = better).
    
    Args:
        reference_palette: Reference dominant colors (N, 4)
        candidate_palettes: Dict mapping path -> dominant colors array
        n_colors: Number of colors to use for EMD
        top_n: Number of results to return
    
    Returns:
        List of (path, distance) tuples, sorted by distance (ascending)
    """
    scores = []
    for path, palette in candidate_palettes.items():
        if palette is not None:
            dist = emd_palette_distance(reference_palette, palette, n_colors=n_colors)
            scores.append((path, dist))
    
    scores.sort(key=lambda x: x[1])
    return scores[:top_n]


def rank_by_moments(
    reference_moments: np.ndarray,
    candidate_moments: dict,
    top_n: int = 20
) -> list:
    """
    Rank candidates by color moments distance (lower = better).
    
    Args:
        reference_moments: Reference moments vector (9,)
        candidate_moments: Dict mapping path -> moments vector
        top_n: Number of results to return
    
    Returns:
        List of (path, distance) tuples, sorted by distance (ascending)
    """
    scores = []
    for path, moments in candidate_moments.items():
        if moments is not None:
            dist = moments_distance(reference_moments, moments)
            scores.append((path, dist))
    
    scores.sort(key=lambda x: x[1])
    return scores[:top_n]


# --------------------------------------------------------------------------------------
# Testing / Demo
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python palette.py <image_path> [image_path_2]")
        print("\nExtracts palette features from image(s).")
        print("If two images provided, computes similarity/distance metrics.")
        sys.exit(1)
    
    path1 = sys.argv[1]
    print(f"\n=== Palette Analysis: {path1} ===\n")
    
    # Extract features
    features1 = extract_all_palette_features(path1)
    
    print(f"Histogram shape: {features1['histogram'].shape}")
    print(f"Dominant colors shape: {features1['dominant'].shape}")
    print(f"Moments shape: {features1['moments'].shape}")
    
    print(f"\nTop 5 dominant colors (LAB + proportion):")
    for i, (l, a, b, p) in enumerate(features1['dominant'][:5]):
        print(f"  {i+1}. L={l:.1f}, A={a:.1f}, B={b:.1f}, proportion={p:.2%}")
    
    print(f"\nColor moments: {features1['moments']}")
    
    if len(sys.argv) >= 3:
        path2 = sys.argv[2]
        print(f"\n=== Comparing with: {path2} ===\n")
        
        features2 = extract_all_palette_features(path2)
        
        hist_sim = histogram_similarity(features1['histogram'], features2['histogram'])
        print(f"Histogram similarity (cosine): {hist_sim:.4f}")
        
        mom_dist = moments_distance(features1['moments'], features2['moments'])
        print(f"Moments distance (euclidean): {mom_dist:.4f}")
        
        for n in [5, 10, 20]:
            emd = emd_palette_distance(features1['dominant'], features2['dominant'], n_colors=n)
            print(f"EMD (top {n} colors): {emd:.4f}")
    
    print(f"\n[POT library available: {HAS_OT}]")

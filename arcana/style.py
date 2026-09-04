# style.py — Style extraction and similarity for Arcana
# Complements palette.py for a palette→style continuum
#
# Methods:
#   - Edge Histogram: Line directions, composition structure (HOG-inspired)
#   - Texture LBP: Local Binary Patterns for surface texture
#   - Gram Matrix: Neural style features from VGG (artistic style)
#
# All methods return fixed-size vectors for cosine similarity comparison.

import numpy as np
try:
    from . import gpu as _gpu
except ImportError:
    import gpu as _gpu

import cv2
try:
    from .cvio import imread_unicode
except ImportError:
    from cvio import imread_unicode
from typing import Tuple, Optional, Union
from functools import lru_cache

# Optional: VGG for Gram matrix
try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None


# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------
DEFAULT_RESIZE_DIM = 256
EDGE_HIST_BINS = 16          # Bins per direction histogram
EDGE_GRID_SIZE = 4           # 4x4 spatial grid = 16 cells
LBP_RADIUS = 1               # LBP neighborhood radius
LBP_POINTS = 8               # LBP sampling points
LBP_HIST_BINS = 26           # Uniform LBP patterns + non-uniform

# VGG layers for Gram matrix style features
# Full: relu1_2 (64ch), relu2_2 (128ch), relu3_3 (256ch), relu4_3 (512ch) → ~174k dims
# Compact: relu2_2, relu3_3 → ~41k dims (still captures style well, 4x smaller)
GRAM_LAYERS = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']
GRAM_LAYERS_COMPACT = ['relu2_2', 'relu3_3']  # 4x smaller, still effective
GRAM_PCA_DIMS = 512  # Target dims for PCA compression (if used)


# --------------------------------------------------------------------------------------
# Image Preprocessing
# --------------------------------------------------------------------------------------
def _load_and_prepare_gray(image_or_path, resize_dim: int = DEFAULT_RESIZE_DIM) -> np.ndarray:
    """
    Load image and convert to grayscale, resized for processing.
    
    Returns:
        Grayscale image as np.ndarray (H, W), dtype uint8
    """
    if isinstance(image_or_path, str):
        img = imread_unicode(image_or_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_or_path}")
    else:
        img = image_or_path
    
    # Resize
    h, w = img.shape[:2]
    scale = resize_dim / max(h, w)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Convert to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    return gray


def _load_and_prepare_rgb(image_or_path, resize_dim: int = DEFAULT_RESIZE_DIM) -> np.ndarray:
    """
    Load image as RGB, resized for processing.
    
    Returns:
        RGB image as np.ndarray (H, W, 3), dtype uint8
    """
    if isinstance(image_or_path, str):
        img = imread_unicode(image_or_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_or_path}")
    else:
        img = image_or_path
    
    # Resize
    h, w = img.shape[:2]
    scale = resize_dim / max(h, w)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # BGR to RGB
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    return img


# --------------------------------------------------------------------------------------
# Method 1: Edge Histogram (HOG-inspired)
# --------------------------------------------------------------------------------------
def extract_edge_histogram(
    image_or_path,
    n_bins: int = EDGE_HIST_BINS,
    grid_size: int = EDGE_GRID_SIZE,
    resize_dim: int = DEFAULT_RESIZE_DIM
) -> np.ndarray:
    """
    Extract edge direction histogram on a spatial grid.
    
    Inspired by HOG but simplified:
    - Compute gradient magnitude and direction
    - Divide image into grid cells
    - For each cell, compute histogram of edge directions weighted by magnitude
    
    Args:
        image_or_path: BGR image or path
        n_bins: Number of orientation bins (0-180 degrees)
        grid_size: NxN spatial grid
        resize_dim: Downscale dimension
    
    Returns:
        Flattened histogram vector of shape (grid_size^2 * n_bins,), dtype float32
    """
    gray = _load_and_prepare_gray(image_or_path, resize_dim)
    
    # Compute gradients (Sobel)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Magnitude and direction
    magnitude = np.sqrt(gx**2 + gy**2)
    direction = np.arctan2(gy, gx) * (180 / np.pi)  # -180 to 180
    direction = (direction + 180) % 180  # 0 to 180 (unsigned gradient)
    
    h, w = gray.shape
    cell_h = h // grid_size
    cell_w = w // grid_size
    
    histograms = []
    for row in range(grid_size):
        for col in range(grid_size):
            # Extract cell
            y0, y1 = row * cell_h, (row + 1) * cell_h
            x0, x1 = col * cell_w, (col + 1) * cell_w
            
            cell_mag = magnitude[y0:y1, x0:x1].flatten()
            cell_dir = direction[y0:y1, x0:x1].flatten()
            
            # Weighted histogram
            hist, _ = np.histogram(cell_dir, bins=n_bins, range=(0, 180), weights=cell_mag)
            
            # Normalize
            hist = hist.astype(np.float32)
            norm = np.linalg.norm(hist)
            if norm > 1e-8:
                hist = hist / norm
            
            histograms.append(hist)
    
    return np.concatenate(histograms).astype(np.float32)


def edge_histogram_similarity(hist_a: np.ndarray, hist_b: np.ndarray) -> float:
    """
    Compute cosine similarity between two edge histograms.
    
    Returns:
        Similarity score in [0, 1], higher = more similar
    """
    norm_a = np.linalg.norm(hist_a)
    norm_b = np.linalg.norm(hist_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(hist_a, hist_b) / (norm_a * norm_b))


# --------------------------------------------------------------------------------------
# Method 2: Texture LBP (Local Binary Patterns)
# --------------------------------------------------------------------------------------
def _compute_lbp(gray: np.ndarray, radius: int = LBP_RADIUS, n_points: int = LBP_POINTS) -> np.ndarray:
    """
    Compute Local Binary Pattern image (vectorized for speed).
    
    For each pixel, compare with n_points neighbors in a circle of given radius.
    The result is a pattern number encoding which neighbors are brighter.
    """
    h, w = gray.shape
    gray = gray.astype(np.float32)
    lbp = np.zeros((h, w), dtype=np.uint8)
    
    # Precompute neighbor offsets
    angles = np.array([2 * np.pi * p / n_points for p in range(n_points)])
    dy = radius * np.sin(angles)
    dx = radius * np.cos(angles)
    
    # For each neighbor direction, compute comparison vectorized
    for p in range(n_points):
        # Compute neighbor coordinates for all pixels
        y_off = dy[p]
        x_off = dx[p]
        
        # Integer and fractional parts for bilinear interpolation
        y0_off = int(np.floor(y_off))
        x0_off = int(np.floor(x_off))
        fy = y_off - y0_off
        fx = x_off - x0_off
        
        # Compute neighbor values using shifted arrays (approximate bilinear)
        # Use simple nearest neighbor for speed
        y_shift = int(round(y_off))
        x_shift = int(round(x_off))
        
        # Create shifted view
        if y_shift >= 0 and x_shift >= 0:
            neighbor = np.zeros_like(gray)
            neighbor[:-y_shift or None, :-x_shift or None] = gray[y_shift:, x_shift:]
        elif y_shift >= 0 and x_shift < 0:
            neighbor = np.zeros_like(gray)
            neighbor[:-y_shift or None, -x_shift:] = gray[y_shift:, :x_shift]
        elif y_shift < 0 and x_shift >= 0:
            neighbor = np.zeros_like(gray)
            neighbor[-y_shift:, :-x_shift or None] = gray[:y_shift, x_shift:]
        else:
            neighbor = np.zeros_like(gray)
            neighbor[-y_shift:, -x_shift:] = gray[:y_shift, :x_shift]
        
        # Compare: if neighbor >= center, set bit
        mask = (neighbor >= gray).astype(np.uint8)
        lbp |= (mask << p)
    
    # Clear border
    lbp[:radius, :] = 0
    lbp[-radius:, :] = 0
    lbp[:, :radius] = 0
    lbp[:, -radius:] = 0
    
    return lbp


def _is_uniform_pattern(pattern: int, n_bits: int = 8) -> bool:
    """Check if LBP pattern is uniform (at most 2 transitions 0→1 or 1→0)."""
    bits = [(pattern >> i) & 1 for i in range(n_bits)]
    transitions = sum(bits[i] != bits[(i + 1) % n_bits] for i in range(n_bits))
    return transitions <= 2


_LBP_BIN_LUT: dict = {}


def _lbp_bin_lut(n_points: int) -> np.ndarray:
    """
    pattern value -> histogram bin, for every pattern that fits in a byte.

    An LBP image is uint8, so there are only 256 possible values however many
    megapixels went in. Classifying each pixel individually is therefore work
    that can be done once per process instead of once per pixel.
    """
    lut = _LBP_BIN_LUT.get(n_points)
    if lut is None:
        size = 1 << n_points
        lut = np.empty(size, dtype=np.int64)
        for pattern in range(size):
            lut[pattern] = (bin(pattern).count("1")
                            if _is_uniform_pattern(pattern, n_points)
                            else n_points + 1)
        _LBP_BIN_LUT[n_points] = lut
    return lut


def _compute_uniform_lbp_histogram(lbp: np.ndarray, n_points: int = LBP_POINTS) -> np.ndarray:
    """
    Compute histogram of uniform LBP patterns.

    Uniform patterns: n_points + 1 bins (one per number of 1s)
    Non-uniform: 1 bin
    Total: n_points + 2 bins

    This used to be a Python loop over lbp.flatten(), calling
    _is_uniform_pattern and bin().count('1') per pixel -- 65,536 iterations for
    a 256px image, run once per cell of a 4x4 grid. It was 436 ms per image and
    87% of the entire style-feature phase, which is why indexing 9,359 pictures
    with style took an hour and a half. The bin for a pattern depends only on
    the pattern, so it is a 256-entry table and a bincount. The output is
    identical.
    """
    lut = _lbp_bin_lut(n_points)
    flat = np.asarray(lbp).ravel()
    if flat.size == 0:
        return np.zeros(n_points + 2, dtype=np.float32)
    counts = np.bincount(lut[flat], minlength=n_points + 2).astype(np.float32)
    total = counts.sum()
    if total > 0:
        counts = counts / total
    return counts


def extract_texture_lbp(
    image_or_path,
    radius: int = LBP_RADIUS,
    n_points: int = LBP_POINTS,
    grid_size: int = 4,
    resize_dim: int = DEFAULT_RESIZE_DIM
) -> np.ndarray:
    """
    Extract Local Binary Pattern texture features on a spatial grid.
    
    Args:
        image_or_path: BGR image or path
        radius: LBP neighborhood radius
        n_points: Number of sampling points
        grid_size: NxN spatial grid
        resize_dim: Downscale dimension
    
    Returns:
        Flattened histogram vector, dtype float32
    """
    gray = _load_and_prepare_gray(image_or_path, resize_dim)
    lbp = _compute_lbp(gray, radius, n_points)
    
    h, w = gray.shape
    cell_h = h // grid_size
    cell_w = w // grid_size
    
    n_bins = n_points + 2  # Uniform patterns + non-uniform
    histograms = []
    
    for row in range(grid_size):
        for col in range(grid_size):
            y0, y1 = row * cell_h, (row + 1) * cell_h
            x0, x1 = col * cell_w, (col + 1) * cell_w
            
            cell_lbp = lbp[y0:y1, x0:x1]
            hist = _compute_uniform_lbp_histogram(cell_lbp, n_points)
            histograms.append(hist)
    
    return np.concatenate(histograms).astype(np.float32)


def texture_lbp_distance(lbp_a: np.ndarray, lbp_b: np.ndarray) -> float:
    """
    Compute chi-squared distance between LBP histograms.
    
    Returns:
        Distance (lower = more similar)
    """
    eps = 1e-10
    chi2 = np.sum((lbp_a - lbp_b) ** 2 / (lbp_a + lbp_b + eps))
    return float(chi2)


def texture_lbp_similarity(lbp_a: np.ndarray, lbp_b: np.ndarray) -> float:
    """
    Compute similarity from LBP histograms (1 / (1 + chi2_distance)).
    
    Returns:
        Similarity in [0, 1], higher = more similar
    """
    dist = texture_lbp_distance(lbp_a, lbp_b)
    return 1.0 / (1.0 + dist)


# --------------------------------------------------------------------------------------
# Method 3: Gram Matrix (Neural Style - VGG)
# --------------------------------------------------------------------------------------
_VGG_MODEL = {}


def _load_vgg():
    """Load VGG19 model for feature extraction (cached)."""
    if not HAS_TORCH:
        raise ImportError("PyTorch required for Gram matrix features")
    
    if "model" not in _VGG_MODEL:
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        vgg.eval()
        
        # Move to GPU if available
        device = torch.device(_gpu.device())
        vgg = vgg.to(device)
        
        # Freeze
        for param in vgg.parameters():
            param.requires_grad = False
        
        _VGG_MODEL["model"] = vgg
        _VGG_MODEL["device"] = device
    
    return _VGG_MODEL["model"], _VGG_MODEL["device"]


def _vgg_preprocess(img_rgb: np.ndarray, device) -> torch.Tensor:
    """Preprocess image for VGG."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = transform(img_rgb).unsqueeze(0)
    return tensor.to(device)


def _compute_gram(features: torch.Tensor) -> torch.Tensor:
    """Compute Gram matrix from feature maps."""
    b, c, h, w = features.shape
    F = features.view(b, c, h * w)
    G = torch.bmm(F, F.transpose(1, 2))  # (b, c, c)
    G = G / (c * h * w)  # Normalize
    return G


# VGG layer indices for style extraction
_VGG_STYLE_LAYERS = {
    'relu1_2': 3,
    'relu2_2': 8,
    'relu3_3': 15,
    'relu4_3': 24,
}


def extract_gram_features(
    image_or_path,
    layers: list = None,
    resize_dim: int = 224,
    compact: bool = False
) -> np.ndarray:
    """
    Extract Gram matrix features from VGG19 layers.
    
    The Gram matrix captures style by measuring correlations between
    feature maps, effectively encoding texture and artistic patterns.
    
    Args:
        image_or_path: BGR image or path
        layers: Which VGG layers to use (default depends on compact flag)
        resize_dim: Input size for VGG (should be ~224)
        compact: If True, use fewer layers (~41k dims instead of ~174k)
    
    Returns:
        Flattened Gram features, dtype float32
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch required for Gram matrix features. Install with: pip install torch torchvision")
    
    if layers is None:
        layers = GRAM_LAYERS_COMPACT if compact else GRAM_LAYERS
    
    vgg, device = _load_vgg()
    
    # Load and preprocess
    img_rgb = _load_and_prepare_rgb(image_or_path, resize_dim)
    x = _vgg_preprocess(img_rgb, device)
    
    # Extract features at each layer
    gram_features = []
    layer_indices = {name: _VGG_STYLE_LAYERS[name] for name in layers}
    max_idx = max(layer_indices.values())
    
    features_at_layer = {}
    for i, layer in enumerate(vgg.children()):
        x = layer(x)
        for name, idx in layer_indices.items():
            if i == idx:
                features_at_layer[name] = x.clone()
        if i >= max_idx:
            break
    
    # Compute Gram matrices
    for name in layers:
        if name in features_at_layer:
            gram = _compute_gram(features_at_layer[name])
            # Flatten upper triangle (Gram is symmetric)
            gram_np = gram.squeeze(0).cpu().numpy()
            upper = gram_np[np.triu_indices(gram_np.shape[0])]
            gram_features.append(upper)
    
    return np.concatenate(gram_features).astype(np.float32)


def extract_gram_features_batch(images, layers: list = None,
                                resize_dim: int = 224, compact: bool = False):
    """
    Gram features for many images, batching every group that shares a shape.

    extract_gram_features runs VGG19 on one image at a time, and the indexing
    loop called it once per file: 9,359 photographs took about ninety minutes at
    ~1.7/s with the device idle between them. CLIP indexing was batched from the
    start; this was not.

    _load_and_prepare_rgb scales the long side and keeps the aspect ratio, so a
    mixed library produces tensors of different shapes that cannot be stacked.
    Rather than forcing everything square -- which would change every feature
    value and quietly break comparison with already-built datasets -- images are
    grouped by shape and each group is run as one batch. A library of uniform
    exports becomes a single pass; a mixed one degrades gracefully toward the
    old per-image cost without ever being wrong.

    Returns a list of feature vectors aligned with `images`.
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch required for Gram matrix features.")
    if layers is None:
        layers = GRAM_LAYERS_COMPACT if compact else GRAM_LAYERS

    vgg, device = _load_vgg()
    layer_indices = {name: _VGG_STYLE_LAYERS[name] for name in layers}
    max_idx = max(layer_indices.values())

    prepared = [_vgg_preprocess(_load_and_prepare_rgb(im, resize_dim), device)
                for im in images]

    groups: dict = {}
    for i, t in enumerate(prepared):
        groups.setdefault(tuple(t.shape[1:]), []).append(i)

    out: list = [None] * len(images)
    for _shape, members in groups.items():
        batch = torch.cat([prepared[i] for i in members], dim=0)
        x = batch
        features_at_layer = {}
        with torch.no_grad():
            for i, layer in enumerate(vgg.children()):
                x = layer(x)
                for name, idx in layer_indices.items():
                    if i == idx:
                        features_at_layer[name] = x.clone()
                if i >= max_idx:
                    break

        per_image = [[] for _ in members]
        for name in layers:
            feat = features_at_layer.get(name)
            if feat is None:
                continue
            b, c, h, w = feat.shape
            flat = feat.reshape(b, c, h * w)
            # Same arithmetic as _compute_gram, done across the batch so the
            # whole thing stays on the device.
            grams = torch.bmm(flat, flat.transpose(1, 2)) / float(c * h * w)
            iu = np.triu_indices(c)
            g = grams.cpu().numpy()
            for k in range(b):
                per_image[k].append(g[k][iu])

        for k, orig_i in enumerate(members):
            out[orig_i] = (np.concatenate(per_image[k]).astype(np.float32)
                           if per_image[k] else np.zeros(0, dtype=np.float32))
    return out


def gram_similarity(gram_a: np.ndarray, gram_b: np.ndarray) -> float:
    """
    Compute cosine similarity between Gram features.
    
    Returns:
        Similarity in [0, 1], higher = more similar
    """
    norm_a = np.linalg.norm(gram_a)
    norm_b = np.linalg.norm(gram_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(gram_a, gram_b) / (norm_a * norm_b))


def gram_distance(gram_a: np.ndarray, gram_b: np.ndarray) -> float:
    """
    Compute L2 distance between Gram features.
    
    Returns:
        Distance (lower = more similar)
    """
    return float(np.linalg.norm(gram_a - gram_b))


# --------------------------------------------------------------------------------------
# Combined Style Extraction
# --------------------------------------------------------------------------------------
def extract_all_style_features(
    image_or_path,
    resize_dim: int = DEFAULT_RESIZE_DIM,
    include_gram: bool = True,
    compact_gram: bool = True
) -> dict:
    """
    Extract all style features for an image.
    
    Args:
        image_or_path: BGR image or path
        resize_dim: Resize dimension
        include_gram: Whether to include Gram features (requires PyTorch)
        compact_gram: If True, use 2 VGG layers (~41k dims) instead of 4 (~174k dims)
    
    Returns:
        Dictionary with keys:
            'edge_histogram': np.ndarray - Edge direction histogram
            'texture_lbp': np.ndarray - LBP texture features
            'gram': np.ndarray - Gram matrix features (if include_gram=True and PyTorch available)
    """
    features = {
        'edge_histogram': extract_edge_histogram(image_or_path, resize_dim=resize_dim),
        'texture_lbp': extract_texture_lbp(image_or_path, resize_dim=resize_dim),
    }
    
    if include_gram and HAS_TORCH:
        try:
            features['gram'] = extract_gram_features(image_or_path, resize_dim=224, compact=compact_gram)
        except Exception as e:
            print(f"[WARN] Gram extraction failed: {e}")
            features['gram'] = None
    else:
        features['gram'] = None
    
    return features


# --------------------------------------------------------------------------------------
# Batch Processing
# --------------------------------------------------------------------------------------
def batch_extract_styles(
    paths: list,
    resize_dim: int = DEFAULT_RESIZE_DIM,
    include_gram: bool = True,
    compact_gram: bool = True,
    show_progress: bool = True
) -> dict:
    """
    Extract style features for a batch of images.
    
    Args:
        paths: List of image paths
        resize_dim: Resize dimension for edge/LBP
        include_gram: Whether to include Gram features
        compact_gram: If True, use 2 VGG layers (~41k dims) instead of 4 (~174k dims)
        show_progress: Show progress bar
    
    Returns:
        Dictionary mapping path -> style_dict
    """
    from tqdm import tqdm
    
    results = {}
    iterator = tqdm(paths, desc="Extracting styles") if show_progress else paths
    
    for path in iterator:
        try:
            results[path] = extract_all_style_features(
                path,
                resize_dim=resize_dim,
                include_gram=include_gram,
                compact_gram=compact_gram
            )
        except Exception as e:
            print(f"[WARN] Failed to extract style for {path}: {e}")
            results[path] = None
    
    return results


# --------------------------------------------------------------------------------------
# Search / Ranking Functions
# --------------------------------------------------------------------------------------
def rank_by_edge_histogram(
    reference: np.ndarray,
    candidates: dict,
    top_n: int = 20
) -> list:
    """
    Rank candidates by edge histogram similarity.
    
    Returns:
        List of (path, similarity) tuples, sorted by similarity (descending)
    """
    scores = []
    for path, feat in candidates.items():
        if feat is not None and feat.get('edge_histogram') is not None:
            sim = edge_histogram_similarity(reference, feat['edge_histogram'])
            scores.append((path, sim))
    
    scores.sort(key=lambda x: -x[1])
    return scores[:top_n]


def rank_by_texture_lbp(
    reference: np.ndarray,
    candidates: dict,
    top_n: int = 20
) -> list:
    """
    Rank candidates by LBP texture similarity.
    
    Returns:
        List of (path, similarity) tuples, sorted by similarity (descending)
    """
    scores = []
    for path, feat in candidates.items():
        if feat is not None and feat.get('texture_lbp') is not None:
            sim = texture_lbp_similarity(reference, feat['texture_lbp'])
            scores.append((path, sim))
    
    scores.sort(key=lambda x: -x[1])
    return scores[:top_n]


def rank_by_gram(
    reference: np.ndarray,
    candidates: dict,
    top_n: int = 20
) -> list:
    """
    Rank candidates by Gram matrix similarity.
    
    Returns:
        List of (path, similarity) tuples, sorted by similarity (descending)
    """
    scores = []
    for path, feat in candidates.items():
        if feat is not None and feat.get('gram') is not None:
            sim = gram_similarity(reference, feat['gram'])
            scores.append((path, sim))
    
    scores.sort(key=lambda x: -x[1])
    return scores[:top_n]


# --------------------------------------------------------------------------------------
# Testing / Demo
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python style.py <image_path> [image_path_2]")
        print("\nExtracts style features from image(s).")
        print("If two images provided, computes similarity metrics.")
        sys.exit(1)
    
    path1 = sys.argv[1]
    print(f"\n=== Style Analysis: {path1} ===\n")
    
    # Extract features
    features1 = extract_all_style_features(path1, include_gram=HAS_TORCH)
    
    print(f"Edge histogram shape: {features1['edge_histogram'].shape}")
    print(f"Texture LBP shape: {features1['texture_lbp'].shape}")
    if features1['gram'] is not None:
        print(f"Gram features shape: {features1['gram'].shape}")
    else:
        print("Gram features: Not available (PyTorch not installed)")
    
    if len(sys.argv) >= 3:
        path2 = sys.argv[2]
        print(f"\n=== Comparing with: {path2} ===\n")
        
        features2 = extract_all_style_features(path2, include_gram=HAS_TORCH)
        
        edge_sim = edge_histogram_similarity(features1['edge_histogram'], features2['edge_histogram'])
        print(f"Edge histogram similarity: {edge_sim:.4f}")
        
        lbp_sim = texture_lbp_similarity(features1['texture_lbp'], features2['texture_lbp'])
        print(f"Texture LBP similarity: {lbp_sim:.4f}")
        
        if features1['gram'] is not None and features2['gram'] is not None:
            gram_sim = gram_similarity(features1['gram'], features2['gram'])
            print(f"Gram matrix similarity: {gram_sim:.4f}")
    
    print(f"\n[PyTorch available: {HAS_TORCH}]")

# classic_transfer.py -- two colour transfers from the papers, written here
#
# MKL, the linear Monge-Kantorovitch transfer (Pitie and Kokaram, CVMP 2007).
# It maps the content's 3x3 colour covariance onto the reference's through the
# symmetric square root, which is the optimal linear map between two Gaussians:
#
#     T = A^-1/2 (A^1/2 B A^1/2)^1/2 A^-1/2,     x' = (x - mu_a) T + mu_b
#
# with A and B the content's and the reference's covariances. Closed form,
# instant, and it matches cross-channel structure that LAB's per-channel mean
# and deviation cannot, which is why it fixes the colour casts LAB leaves.
#
# IDT, iterative N-dimensional PDF transfer (Pitie, Kokaram and Dahyot, ICCV
# 2005 and CVIU 2007). Each iteration rotates both colour clouds at random,
# matches the content to the reference along each rotated axis with a 1-D
# histogram match, and rotates back. After a dozen rotations the whole 3-D
# distribution matches, not only its first two moments. It runs on a working
# size and reaches full resolution through a 3-D LUT fitted on the before and
# after pair, the same way ModFlows does, and the LUT's smoothing does the work
# of the CVIU paper's grain suppression.
#
# Both work on RGB in 0..1 and take a strength that blends the result with the
# original. python-color-transfer (MIT) implements both and served as the
# check on behaviour. This code is our own.

from __future__ import annotations

import numpy as np
from PIL import Image

WORK_SIDE = 384          # IDT's working size; the LUT carries it to full size
STATS_SIDE = 512         # MKL's covariances are estimated at this size
IDT_ITERATIONS = 12
EPS = 1e-6


def _pixels(img: Image.Image, side: int) -> np.ndarray:
    """An image's colours as N x 3 floats in 0..1, from a copy at most `side` across."""
    small = img.convert("RGB")
    if max(small.size) > side:
        small = small.copy()
        small.thumbnail((side, side), Image.BILINEAR)
    return np.asarray(small, np.float64).reshape(-1, 3) / 255.0


def _sqrtm(m: np.ndarray, inverse: bool = False) -> np.ndarray:
    """The symmetric square root (or its inverse) of a symmetric positive matrix."""
    w, v = np.linalg.eigh(m)
    w = np.maximum(w, EPS)
    d = 1.0 / np.sqrt(w) if inverse else np.sqrt(w)
    return (v * d) @ v.T


def mkl_matrix(content: np.ndarray, reference: np.ndarray):
    """The MKL map between two N x 3 colour clouds: (T, mean_a, mean_b)."""
    mu_a, mu_b = content.mean(0), reference.mean(0)
    a = np.cov(content, rowvar=False) + EPS * np.eye(3)
    b = np.cov(reference, rowvar=False) + EPS * np.eye(3)
    a_half, a_inv_half = _sqrtm(a), _sqrtm(a, inverse=True)
    t = a_inv_half @ _sqrtm(a_half @ b @ a_half) @ a_inv_half
    return t, mu_a, mu_b


def mkl(content: Image.Image, reference: Image.Image, strength: float = 1.0) -> Image.Image:
    """The content recoloured toward the reference by the linear MK map, at full size."""
    t, mu_a, mu_b = mkl_matrix(_pixels(content, STATS_SIDE), _pixels(reference, STATS_SIDE))
    full = np.asarray(content.convert("RGB"), np.float32) / 255.0
    flat = full.reshape(-1, 3)
    out = (flat - mu_a.astype(np.float32)) @ t.astype(np.float32) + mu_b.astype(np.float32)
    return _finish(flat, out, strength, full.shape)


def _match_1d(x: np.ndarray, y_sorted: np.ndarray) -> np.ndarray:
    """Values of x moved to the same quantiles of y (a 1-D histogram match)."""
    order = np.argsort(x, kind="stable")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(x))
    q = (ranks + 0.5) / len(x)
    yq = (np.arange(len(y_sorted)) + 0.5) / len(y_sorted)
    return np.interp(q, yq, y_sorted)


def _rotation(rng: np.random.Generator) -> np.ndarray:
    q, r = np.linalg.qr(rng.standard_normal((3, 3)))
    return q * np.sign(np.diag(r))


def idt_cloud(content: np.ndarray, reference: np.ndarray, iterations: int = IDT_ITERATIONS,
              seed: int = 0) -> np.ndarray:
    """The content's N x 3 colours moved so their 3-D distribution matches the reference's."""
    rng = np.random.default_rng(seed)
    x = content.copy()
    for _ in range(iterations):
        r = _rotation(rng)
        xp, yp = x @ r, reference @ r
        moved = np.empty_like(xp)
        for d in range(3):
            moved[:, d] = _match_1d(xp[:, d], np.sort(yp[:, d]))
        x = x + (moved - xp) @ r.T
    return x


def idt(content: Image.Image, reference: Image.Image, strength: float = 1.0,
        iterations: int = IDT_ITERATIONS, lut_n: int = 33) -> Image.Image:
    """The content recoloured by IDT at the working size, carried to full size by a LUT."""
    try:
        from .color_transfer import _apply_lut3d, _fit_lut3d
    except ImportError:                               # pragma: no cover
        from color_transfer import _apply_lut3d, _fit_lut3d
    full = content.convert("RGB")
    low = full.copy()
    low.thumbnail((WORK_SIDE, WORK_SIDE), Image.BILINEAR)
    src = np.asarray(low, np.float64).reshape(-1, 3) / 255.0
    moved = idt_cloud(src, _pixels(reference, WORK_SIDE), iterations)
    low_after = Image.fromarray((np.clip(moved, 0, 1) * 255).round().astype(np.uint8)
                                .reshape(np.asarray(low).shape))
    out = _apply_lut3d(full, _fit_lut3d(low, low_after, lut_n))
    if strength >= 1.0:
        return out
    return Image.blend(full, out, float(max(0.0, strength)))


def _finish(flat: np.ndarray, out: np.ndarray, strength: float, shape) -> Image.Image:
    s = float(np.clip(strength, 0.0, 1.0))
    mixed = flat * (1.0 - s) + out * s
    return Image.fromarray((np.clip(mixed, 0, 1) * 255).round().astype(np.uint8).reshape(shape))

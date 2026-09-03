# refselect.py -- choosing which colours of a reference image actually get used
#
# Both colour-transfer methods take a reference *image* and read its colour
# distribution: the LAB method takes its per-channel mean and standard
# deviation, ModFlows encodes it. Neither offers any way to say "use the
# turquoise highlights, not the black background".
#
# That is a real problem, because photographs are rarely evenly distributed. A
# picture that reads as iridescent teal can be 70% near-black, and both methods
# will faithfully transfer mostly-black -- the result looks muddy and the user
# concludes the transfer is broken when it is doing exactly what it was asked.
#
# The fix does not need to touch either method. If the reference is filtered
# down to the pixels you actually meant, every downstream method sees the
# distribution you wanted. This module does that filtering and packs the
# survivors into a small proxy image that can be passed anywhere the original
# reference could.
#
# The two controls are chosen because they match how people describe the
# problem:
#
#   lightness (L* in CIELAB, 0..100)  "ignore the black background"
#   saturation (S in HSV, 0..100)     "ignore the grey concrete"
#
# Both are ranges rather than floors, so "only the shadows" is as expressible
# as "only the highlights".

from __future__ import annotations

import numpy as np
from PIL import Image

# Sampling cap. A 24 MP reference has ~24M pixels; the statistics of a 512x512
# sample are indistinguishable for this purpose and 90x cheaper.
SAMPLE_SIDE = 512

# Below this fraction of surviving pixels the estimate stops being meaningful
# and the caller should say so rather than silently transferring noise.
MIN_KEEP = 0.002


def _to_rgb_array(img: Image.Image, max_side: int = SAMPLE_SIDE) -> np.ndarray:
    im = img.convert("RGB")
    if max(im.size) > max_side:
        im = im.copy()
        im.thumbnail((max_side, max_side), Image.LANCZOS)
    return np.asarray(im, dtype=np.uint8)


def _srgb_to_lab_l(rgb: np.ndarray) -> np.ndarray:
    """
    CIELAB L* for an (N, 3) uint8 array, as 0..100.

    Written out rather than pulled from skimage/cv2 so this module stays a
    numpy-and-Pillow dependency: it is imported by the UI on every slider move.
    """
    srgb = rgb.astype(np.float32) / 255.0
    lin = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
    # Y row of the sRGB D65 matrix -- L* only needs luminance.
    y = lin @ np.array([0.2126729, 0.7151522, 0.0721750], dtype=np.float32)
    eps = (6.0 / 29.0) ** 3
    fy = np.where(y > eps, np.cbrt(y), y / (3 * (6.0 / 29.0) ** 2) + 4.0 / 29.0)
    return 116.0 * fy - 16.0


def _hsv_saturation(rgb: np.ndarray) -> np.ndarray:
    """HSV S for an (N, 3) uint8 array, as 0..100."""
    v = rgb.max(axis=1).astype(np.float32)
    c = v - rgb.min(axis=1).astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(v > 0, c / v, 0.0)
    return s * 100.0


def analyse(img: Image.Image) -> dict:
    """Lightness and saturation distribution of a reference, for the UI."""
    px = _to_rgb_array(img).reshape(-1, 3)
    lightness = _srgb_to_lab_l(px)
    saturation = _hsv_saturation(px)
    return {
        "n": int(px.shape[0]),
        "lightness": lightness,
        "saturation": saturation,
        "rgb": px,
    }


def keep_mask(px: np.ndarray, l_min: float = 0.0, l_max: float = 100.0,
              s_min: float = 0.0, s_max: float = 100.0) -> np.ndarray:
    """Boolean mask over an (N, 3) uint8 pixel array."""
    lightness = _srgb_to_lab_l(px)
    saturation = _hsv_saturation(px)
    return ((lightness >= l_min) & (lightness <= l_max)
            & (saturation >= s_min) & (saturation <= s_max))


def filter_reference(img: Image.Image, l_min: float = 0.0, l_max: float = 100.0,
                     s_min: float = 0.0, s_max: float = 100.0) -> tuple[Image.Image, float]:
    """
    Build a proxy reference containing only the pixels in range.

    Returns (proxy image, fraction of pixels kept). The proxy is a square image
    made of the surviving pixels in their original proportions -- it looks like
    noise, but every method here only ever reads the colour *distribution*, and
    the distribution is exactly right.

    When the filter is wide open the original image is returned untouched, so
    the default path is bit-for-bit what it was before this module existed.
    """
    wide_open = (l_min <= 0 and l_max >= 100 and s_min <= 0 and s_max >= 100)
    if wide_open:
        return img, 1.0

    px = _to_rgb_array(img).reshape(-1, 3)
    mask = keep_mask(px, l_min, l_max, s_min, s_max)
    kept = px[mask]
    frac = float(kept.shape[0]) / float(px.shape[0]) if px.shape[0] else 0.0
    if kept.shape[0] == 0:
        return img, 0.0

    # Pack into a square. Tiling to fill the last row keeps the proportions of
    # the survivors intact -- sampling with replacement would too, but tiling is
    # deterministic, which matters when the user nudges a slider and expects the
    # preview to change only because the filter changed.
    side = max(8, int(np.sqrt(kept.shape[0])))
    need = side * side
    if kept.shape[0] < need:
        reps = int(np.ceil(need / kept.shape[0]))
        kept = np.tile(kept, (reps, 1))
    kept = kept[:need].reshape(side, side, 3)
    return Image.fromarray(kept.astype(np.uint8), "RGB"), frac


def palette_strip(img: Image.Image, n: int = 12, l_min: float = 0.0,
                  l_max: float = 100.0, s_min: float = 0.0,
                  s_max: float = 100.0,
                  source_path: str | None = None) -> list[tuple[str, float]]:
    """
    The dominant colours that survive the filter, as ('#rrggbb', share) pairs
    sorted darkest first, where share sums to 1.

    This deliberately uses palette.extract_dominant_colors -- the same k-means
    in LAB space that the /palette endpoint draws under the Reference and
    Target thumbnails. An earlier version used fast uniform RGB quantisation
    and took the most *frequent* bins, which on a mostly-dark photograph
    returned fourteen near-identical dark swatches: it looked like the strip
    was showing only one end of the palette, and it disagreed with the strip
    directly below it in the same panel. Two strips describing the same picture
    have to agree, so there is now one algorithm.

    Callers draw each swatch at a width proportional to its share, which is
    also what /palette does -- so a picture that is 70% near-black shows a wide
    dark band, and raising l_min visibly reclaims that width for the colours
    you actually want.
    """
    wide_open = (l_min <= 0 and l_max >= 100 and s_min <= 0 and s_max >= 100)
    px = _to_rgb_array(img, max_side=256).reshape(-1, 3)
    mask = keep_mask(px, l_min, l_max, s_min, s_max)
    kept = px[mask]
    if kept.shape[0] == 0:
        return []

    # k-means needs at least as many distinct samples as clusters.
    n_eff = int(min(n, len(np.unique(kept, axis=0))))
    if n_eff < 1:
        return []
    if n_eff == 1:
        r, g, b = kept[0]
        return [(f"#{r:02x}{g:02x}{b:02x}", 1.0)]

    try:
        from .palette import extract_dominant_colors
    except ImportError:
        from palette import extract_dominant_colors

    if wide_open:
        # Hand the untouched image over, so an unfiltered strip is bit-for-bit
        # what /palette draws under the thumbnails. Sampling to 256 and then
        # repacking into a square feeds k-means a different pixel population
        # and a different ordering, which was enough to shift every swatch by a
        # few values -- close enough to look like a bug, far enough to be one.
        #
        # Passing the path rather than the decoded pixels closes the last gap:
        # extract_dominant_colors then reads the file with OpenCV exactly as
        # /palette does, and Pillow and OpenCV disagree by +-1 per channel on
        # JPEG (different IDCT), which is invisible but still not "identical".
        if source_path:
            return _finish(extract_dominant_colors(source_path, n_colors=n_eff))
        source = np.asarray(img.convert("RGB"))
    else:
        # Filtered: the survivors are a scattered subset, so they have to be
        # packed back into a rectangle before an image-shaped API can see them.
        side = max(2, int(np.sqrt(kept.shape[0])))
        need = side * side
        block = kept
        if block.shape[0] < need:
            block = np.tile(block, (int(np.ceil(need / block.shape[0])), 1))
        source = block[:need].reshape(side, side, 3)

    # extract_dominant_colors takes BGR, and returns [L, A, B, proportion].
    return _finish(extract_dominant_colors(source[:, :, ::-1].copy(), n_colors=n_eff))


def _finish(pal) -> list[tuple[str, float]]:
    """LAB palette rows -> ('#rrggbb', share) pairs, darkest first."""
    out = []
    for l_star, a, b, prop in pal:
        rgb = _lab_to_srgb(float(l_star), float(a), float(b))
        out.append((float(l_star), f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}", float(prop)))
    out.sort(key=lambda t: t[0])
    total = sum(p for _l, _h, p in out) or 1.0
    return [(hex_, p / total) for _l, hex_, p in out]


def _lab_to_srgb(l_star: float, a: float, b: float) -> tuple[int, int, int]:
    """CIELAB (L* 0..100, a/b roughly -128..127) to 8-bit sRGB, D65."""
    fy = (l_star + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0
    delta = 6.0 / 29.0

    def finv(t):
        return t ** 3 if t > delta else 3 * delta * delta * (t - 4.0 / 29.0)

    # D65 white point
    x = 0.95047 * finv(fx)
    y = 1.00000 * finv(fy)
    z = 1.08883 * finv(fz)

    r = x * 3.2404542 + y * -1.5371385 + z * -0.4985314
    g = x * -0.9692660 + y * 1.8760108 + z * 0.0415560
    bl = x * 0.0556434 + y * -0.2040259 + z * 1.0572252

    def gamma(c):
        c = max(0.0, min(1.0, c))
        return 1.055 * (c ** (1 / 2.4)) - 0.055 if c > 0.0031308 else 12.92 * c

    return tuple(int(round(gamma(c) * 255)) for c in (r, g, bl))

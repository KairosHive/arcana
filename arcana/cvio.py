# cvio.py — OpenCV file IO that survives non-ASCII paths
#
# cv2.imread and cv2.imwrite pass the filename to the C++ layer as bytes in the
# system locale encoding. On Windows that is a legacy code page (cp1252 here), so
# any path containing a character outside it -- "café", "señor", "日本", most
# non-English photo libraries -- simply fails. imread returns None and imwrite
# returns False, neither of which raises, so images vanish from an index with
# nothing but a warning line.
#
# Reading the bytes ourselves and handing OpenCV a buffer sidesteps the encoding
# entirely: Python opens the file with the correct wide-character API.

from __future__ import annotations

import os

import cv2
import numpy as np


def imread_unicode(path: str, flags: int = cv2.IMREAD_COLOR):
    """
    cv2.imread that works for any path Python can open.

    Returns the decoded image, or None if the file is missing or not decodable --
    matching cv2.imread's contract so call sites need no other change.
    """
    try:
        with open(path, "rb") as f:
            buf = np.frombuffer(f.read(), dtype=np.uint8)
    except OSError:
        return None
    if buf.size == 0:
        return None
    try:
        return cv2.imdecode(buf, flags)
    except cv2.error:
        return None


def imwrite_unicode(path: str, img, params: list[int] | None = None) -> bool:
    """
    cv2.imwrite that works for any path Python can create.

    Encodes in memory, then writes the bytes. Returns True on success.
    """
    ext = os.path.splitext(path)[1]
    if not ext:
        ext = ".png"
    try:
        ok, buf = cv2.imencode(ext, img, params or [])
    except cv2.error:
        return False
    if not ok:
        return False
    try:
        with open(path, "wb") as f:
            f.write(buf.tobytes())
    except OSError:
        return False
    return True


# The encoders all resize to 224x224, so decoding a 24-megapixel photograph at
# full resolution and then throwing 99.9% of the pixels away is the single most
# expensive thing indexing does. Measured over 64 real photographs (4000x6000),
# per image, end to end:
#
#     full decode                       187.2 ms   (169.6 of it in the resize)
#     1/4-scale decode                   21.2 ms   -> 8.8x
#
# JPEG's DCT structure makes reduced-scale decoding nearly free -- libjpeg skips
# coefficients rather than decoding and downsampling. The cost is a slightly
# different resampling path, measured against full-resolution embeddings on the
# same 48 photographs:
#
#     1/2  mean cosine 0.9996, nearest-neighbour agreement 98%
#     1/4  mean cosine 0.9994, nearest-neighbour agreement 96%
#     1/8  mean cosine 0.9978, nearest-neighbour agreement 94%
#
# 1/4 is the chosen default: the vectors are the same to four decimal places and
# the neighbour disagreements are near-ties. Anything below MIN_SIDE would start
# feeding the encoder an image smaller than its own input, which is a real loss
# rather than a rounding difference, so small pictures step back up.
ENCODER_MIN_SIDE = 256


def imread_for_encoder(path: str, min_side: int = ENCODER_MIN_SIDE):
    """
    Decode an image for a vision encoder, as cheaply as the picture allows.

    Tries progressively less aggressive reductions until the result is at least
    `min_side` on its short edge, so a small image is never upscaled into the
    encoder. Returns None on an unreadable file, like imread_unicode.
    """
    for flag in (cv2.IMREAD_REDUCED_COLOR_4,
                 cv2.IMREAD_REDUCED_COLOR_2,
                 cv2.IMREAD_COLOR):
        im = imread_unicode(path, flag)
        if im is None:
            return None                     # unreadable; a smaller flag will not help
        if min(im.shape[:2]) >= min_side or flag == cv2.IMREAD_COLOR:
            return im
    return None

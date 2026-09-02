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

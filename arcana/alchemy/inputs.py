"""The pictures an Alchemy step reads, and the pictures it makes.

A picture is named in one of three ways:

    japan:1423        a picture in a dataset, by handle
    out:3f9c1e        a derived picture, by its ledger handle (ledger.py)
    "D:/photos/x.jpg" a file, as a collection holds them (for the person and the terminal; an
                      agent names pictures by handle)

Every picture is read once at each frame size and kept, bounded by bytes, so a slider that redraws
twenty times a second never decodes a 6000 px JPEG twenty times. What a step keeps is written to
output/edited/<family>/<date>/ as <picture>__<method>__<time>.png, and `alchemy make` records it in
the ledger, which gives it its out: handle.
"""

from __future__ import annotations

import collections
import datetime
import json
import os
import threading

import numpy as np
from PIL import Image, ImageOps

from . import pixels as cx
from .memory import MODELS

FRAMES_BYTES = 900_000_000
_frames: collections.OrderedDict = collections.OrderedDict()
_frames_lock = threading.Lock()
MODELS.add("alchemy frames", _frames.clear, _frames_lock)


def _paths():
    try:
        from .. import paths
    except ImportError:                                   # pragma: no cover
        import paths
    return paths


def resolve(ref: str) -> str:
    """A picture's file from its handle (japan:1423), its out: handle or its path, through the one
    resolver every op uses (ops.archive.resolve_item). Ops have already applied the caller's rules
    (an agent names pictures by handle), so this resolves as the person."""
    from ..ops import archive

    if not (ref or "").strip():
        raise ValueError("No picture given.")
    item = archive.resolve_item(str(ref).strip().strip('"'), "person")
    if not os.path.isfile(item.path):
        raise ValueError(f"{ref}'s file is gone: {item.path}")
    return item.path


_sizes: dict = {}


def original_size(path):
    """(width, height) as the picture is shown, after its EXIF rotation. Read from the header and
    kept per file and modification time: a preview asks for it every time, and decoding a 6000 px
    JPEG to learn its size cost more than the whole rest of a finish change."""
    key = (str(path), os.path.getmtime(path))
    got = _sizes.get(key)
    if got is None:
        with Image.open(path) as im:
            w, h = im.size
            turned = im.getexif().get(0x0112, 1) in (5, 6, 7, 8)   # orientations that swap the sides
        got = _sizes[key] = (h, w) if turned else (w, h)
    return got


def frame(path, long_side, multiple=8):
    """(W, H): the picture's own shape with `long_side` pixels on its long side."""
    w, h = original_size(path)
    k = long_side / max(w, h)
    W = max(multiple, int(round(w * k / multiple)) * multiple)
    H = max(multiple, int(round(h * k / multiple)) * multiple)
    return W, H


def load_frame(path, width, height):
    """A photograph covering a W x H frame (cropped to its shape, centred), floats in 0..1, kept."""
    key = (str(path), int(width), int(height))
    MODELS.used("alchemy frames")
    with _frames_lock:
        if key in _frames:
            _frames.move_to_end(key)
            return _frames[key]
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.draft("RGB", (int(width) * 2, int(height) * 2))
        got = np.asarray(cx.cover(im, int(width), int(height)), np.float32) / 255.0
    got.setflags(write=False)
    if got.nbytes < FRAMES_BYTES // 4:
        with _frames_lock:
            _frames[key] = got
            while len(_frames) > 1 and sum(a.nbytes for a in _frames.values()) > FRAMES_BYTES:
                _frames.popitem(last=False)
    return got


def crop_of(path, W, H, zoom=1.0, seed=0):
    """A W x H piece of a material photograph from a copy `zoom` times the frame's size: the middle,
    or somewhere else for another seed."""
    big = load_frame(path, max(W, round(W * zoom)), max(H, round(H * zoom)))
    rng = np.random.default_rng(int(seed))
    x = (big.shape[1] - W) // 2 if not seed else int(rng.integers(0, big.shape[1] - W + 1))
    y = (big.shape[0] - H) // 2 if not seed else int(rng.integers(0, big.shape[0] - H + 1))
    return big[y:y + H, x:x + W]


def to_image(a) -> Image.Image:
    return Image.fromarray((np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8))


def _slug(text: str, n: int = 40) -> str:
    keep = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in str(text))
    return "-".join(p for p in keep.split("-") if p)[:n] or "picture"


def write(image: Image.Image, family: str, title: str, provenance: dict) -> str:
    """A kept picture written to output/edited/<family>/<date>/<title>__<time>.png, its provenance
    inside the PNG so it survives the file being copied anywhere. Returns the path. The ledger
    (ledger.py) records it and gives it its out: handle."""
    from PIL import PngImagePlugin

    folder = _paths().edited_dir(family)
    stem = "__".join(_slug(t) for t in title.split("__") if t) or "kept"
    path = os.path.join(folder, f"{stem}__{datetime.datetime.now().strftime('%H%M%S')}.png")
    n = 2
    while os.path.exists(path):
        path = os.path.join(folder, f"{stem}__{datetime.datetime.now().strftime('%H%M%S')}-{n}.png")
        n += 1
    meta = PngImagePlugin.PngInfo()
    for k, v in provenance.items():
        meta.add_text(f"arcana:{k}", v if isinstance(v, str) else json.dumps(v, default=str))
    image.save(path, "PNG", pnginfo=meta)
    return path

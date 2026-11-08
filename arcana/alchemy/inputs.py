"""The pictures an Alchemy step reads, and the pictures it makes.

A picture is named in one of three ways:

    japan:1423        a picture in a dataset, by handle
    out:3f9c1e2a      a picture an earlier Alchemy step made (in output/alchemy)
    "D:/photos/x.jpg" a file, as the collection rail holds them (uploads included); it must be inside
                      the media roots or the data directory

Every picture is read once at each frame size and kept, bounded by bytes, so a slider that redraws
twenty times a second never decodes a 6000 px JPEG twenty times. What a step makes is written to
output/alchemy as a PNG with a JSON sidecar beside it: its id, the op and arguments that made it, the
pictures it came from, and when. That is the record the output ledger (OPS_PLAN.md, P1) reads, so a
made picture gets its `out:` handle there with no change here.
"""

from __future__ import annotations

import collections
import datetime
import hashlib
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


def out_dir() -> str:
    p = _paths()
    return p.ensure_dir(os.path.join(p.subdir("output"), "alchemy"))


def resolve(ref: str) -> str:
    """A picture's file from its handle, its out: id or its path."""
    ref = (ref or "").strip().strip('"')
    if not ref:
        raise ValueError("No picture given.")
    if ref.startswith("out:"):
        oid = ref[4:]
        if not oid.replace("-", "").isalnum():
            raise ValueError(f"{ref!r} is not an Alchemy output id.")
        path = os.path.join(out_dir(), oid + ".png")
        if not os.path.isfile(path):
            raise ValueError(f"No Alchemy output called {ref!r}.")
        return path
    head, sep, tail = ref.rpartition(":")
    if sep and tail.isdigit() and head and not (len(head) == 1 and head.isalpha()):
        from ..ops import archive

        ds_id, ds, keys = archive.resolve_handles([ref])
        path = ds.idx2path[keys[0]]
        if not os.path.isfile(path):
            raise ValueError(f"{ref}'s file is gone: {path}")
        return path
    path = os.path.abspath(ref)
    paths = _paths()
    if not os.path.isfile(path):
        raise ValueError(f"No picture at {ref!r}.")
    if not (handle_of(path) or paths.is_within(path) or paths.is_within(path, [paths.data_dir()])):
        raise ValueError(f"{ref!r} is neither a picture of a dataset nor inside the media roots; "
                         "add its folder as a dataset first.")
    return path


_members: dict = {}
_members_at = [0.0]
_members_lock = threading.Lock()


def handle_of(path: str) -> str | None:
    """The handle of a file that belongs to a dataset (the collection rail holds files), or None.
    The datasets' file lists are read at most once a minute."""
    import time

    key = os.path.normcase(os.path.abspath(path))
    with _members_lock:
        if time.monotonic() - _members_at[0] > 60:
            _members.clear()
            try:
                from .. import core
                from ..ops import archive
            except ImportError:                           # pragma: no cover
                return None
            for d in core.find_datasets():
                if d.get("modality", "image") != "image":
                    continue
                try:
                    ds_id = archive.dataset_id(d["name"], d["modality"])
                    ds = archive.open_dataset(ds_id)
                except Exception:
                    continue
                for k, p in ds.idx2path.items():
                    _members[os.path.normcase(os.path.abspath(p))] = archive.handle(ds_id, k)
            _members_at[0] = time.monotonic()
        return _members.get(key)


def original_size(path):
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        return im.size


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


def save(image: Image.Image, op: str, args: dict, sources: list[str]) -> tuple[str, str]:
    """A made picture written to output/alchemy with its sidecar: (out id, path)."""
    stamp = datetime.datetime.now().isoformat(timespec="seconds")
    oid = hashlib.blake2b(json.dumps([op, args, sources, stamp], sort_keys=True, default=str).encode(),
                          digest_size=5).hexdigest()
    path = os.path.join(out_dir(), oid + ".png")
    image.save(path)
    with open(os.path.join(out_dir(), oid + ".json"), "w", encoding="utf-8") as f:
        json.dump({"id": "out:" + oid, "path": path, "op": op, "args": args, "sources": sources,
                   "kind": "derived", "made": stamp}, f, indent=1, default=str)
    return "out:" + oid, path

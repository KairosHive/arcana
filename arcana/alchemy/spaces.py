"""The token spaces Alchemy mixes in, and Real-ESRGAN.

- DINOv2   DINOv2-B with registers, drawn back by RAE's ViT-XL decoder (MIT). Two materials mixed
           here become one new surface: the decoder redraws, so a mix never reads as a double
           exposure, at the cost of the photograph's exact grain.
- FLUX.2 VAE  FLUX.2 klein's autoencoder (Apache 2.0), 2 x 2 latents packed into one token per 16 px.
           It keeps a photograph almost exactly, so a material keeps its own grain; a mix is settled
           channel by channel and reads as layers rather than one new surface.

Both share one grid, one token per 16 x 16 pixels, so every token method runs on either. Each loads
on first use under the memory manager. Their principal axes over a sample of the person's own
archive are worked out once and kept in the data directory, and four of them are named by what they
do to a picture: crispness, warmth, exposure and contrast.
"""

from __future__ import annotations

try:
    from arcana import hub as _hub
except ImportError:                                   # pragma: no cover
    import hub as _hub

import os
import pathlib
import random
import threading

import numpy as np

from .memory import LOAD, MODELS

RAE_REPO = "nyu-visionx/RAE-collections"
RAE_FILE = "decoders/dinov2/wReg_base/ViTXL_n08_i512/model.pt"
ESRGAN_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
SPACES = {"DINOv2": "one new surface: the decoder redraws, so materials fuse",
          "FLUX.2 VAE": "the photographs' own grain: a mix reads as layers"}

_held: dict = {}
_lock = threading.RLock()        # one frame at a time on the decoders, from any thread


def device() -> str:
    try:
        from .. import gpu as _gpu
    except ImportError:                                   # pragma: no cover
        import gpu as _gpu
    return _gpu.device()


def lock():
    return _lock


def _subdir(name: str) -> str:
    try:
        from .. import paths
    except ImportError:                                   # pragma: no cover
        import paths
    return paths.ensure_dir(os.path.join(paths.subdir(name), "alchemy"))


def space(name: str):
    """A token space by name, loaded on first use: grid, encode, decode, shell (and, for the VAE,
    settle)."""
    if name not in SPACES:
        raise ValueError(f"No token space called {name!r}. Spaces: {', '.join(SPACES)}.")
    key = "tokens: " + name
    with LOAD:
        got = _held.get(name)
        if got is None:
            with MODELS.loading(key, lambda: _held.pop(name, None), _lock, guess=1.0 if name == "FLUX.2 VAE" else 2.0):
                if name == "FLUX.2 VAE":
                    from .fluxvae import FluxVAE

                    got = FluxVAE(device())
                else:
                    from huggingface_hub import hf_hub_download

                    from .rae import load

                    got = load(_hub.pretrained(hf_hub_download, RAE_REPO, RAE_FILE), device())
            _held[name] = got
            print(f"  alchemy: {key} up", flush=True)
    MODELS.used(key)
    return got


def _esrgan_weights() -> str:
    path = os.path.join(_subdir("models"), "RealESRGAN_x4plus.pth")
    if not os.path.isfile(path):
        import shutil
        import urllib.request

        part = path + ".part"
        with urllib.request.urlopen(ESRGAN_URL, timeout=120) as r, open(part, "wb") as out:
            shutil.copyfileobj(r, out)
        os.replace(part, path)
    return path


def enlarge(image, keep=False):
    """A picture ([h, w, 3] floats) four times larger by Real-ESRGAN. `keep`: a tensor on the card."""
    from . import upscale

    key = "enlarge: Real-ESRGAN"
    with LOAD:
        net = _held.get("esrgan")
        if net is None:
            with MODELS.loading(key, lambda: _held.pop("esrgan", None), _lock, guess=1.0):
                net = _held["esrgan"] = upscale.load(_esrgan_weights(), "cuda" if device().startswith("cuda") else "cpu")
    MODELS.used(key)
    with _lock:
        return upscale.enlarge(net, image, keep) if keep else upscale.enlarge_tiled(net, image)


# ─────────────────────────── principal axes and the four looks ───────────────────────────

def _archive_sample(n=48):
    """Up to `n` photographs from the person's datasets, the same ones every time."""
    try:
        from .. import core
    except ImportError:                                   # pragma: no cover
        import core
    found = []
    for d in core.find_datasets():
        if d.get("modality", "image") != "image":
            continue
        try:
            ds = core.open_dataset(d["name"], d["modality"], 2 if 2 in (d.get("layouts") or [2]) else d["layouts"][0])
        except Exception:
            continue
        found += [p for p in ds.idx2path.values() if os.path.isfile(p)]
        if len(found) > 4000:
            break
    random.Random(0).shuffle(found)
    return found[:n]


_axes: dict = {}


def axes(name: str, sp):
    """(mean, basis, spread) of a space's tokens over a sample of the archive, and which of the first
    axes are crispness, warmth, exposure and contrast."""
    from . import fluxvae
    from .inputs import load_frame

    if name in _axes:
        return _axes[name]
    cache = pathlib.Path(_subdir("models")) / f"axes-{name.replace(' ', '-').replace('.', '')}.npz"
    pics = _archive_sample()
    if not pics and not cache.is_file():
        raise ValueError("The look controls need at least one dataset to learn the token axes from.")
    frames = [load_frame(p, 512, 336) for p in pics]
    ax = fluxvae.axes(sp, frames, str(cache))
    probe = frames[0] if frames else None
    ctl = fluxvae.controls(sp, ax, probe) if probe is not None else {}
    _axes[name] = (ax, ctl)
    return _axes[name]

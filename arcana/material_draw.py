# material_draw.py -- new material drawn from nothing, from places you choose in your archive
#
# The Generate tab's engine. A material is drawn from SOURCES, each with a weight:
#
#   a set          its IP label (made from the set the first time it is asked for)
#   an IP label    a place measured on your pictures, a set, words matched in the archive, one picture
#   a picture      one photograph's own place
#
# plus, optionally, words. Two ways to draw, the same two the material editor's swatch has:
#
#   the place (SD 1.5)     Stable Diffusion with its IP-Adapter reads the sources' blended position in
#                          IP space. It mixes evenly, pushes past what the archive holds (push), slides
#                          toward one member's own look (spread), walks between two places, and draws
#                          tileable plates. Measured: it draws the archive's materials a little brighter
#                          and more graphic than the photographs.
#   the pictures (FLUX)    FLUX.2 klein looks at a few of the sources' own photographs (near each
#                          label's centre, unlike each other, shared out by weight) and draws one material
#                          from them. Faithful to the photographs; uneven when sources are mixed (shown
#                          lichen and water, it drew almost all lichen).
#
# Drawn pictures are drafts in <data dir>/cache/generate until kept: `keep` writes one to
# output/generated/materials/<date>/ and records it in the ledger as generated, so it never passes for
# a photograph of the archive.

from __future__ import annotations

import contextlib
import datetime
import hashlib
import json
import os
import time

import numpy as np

ENGINES = ("the place (SD 1.5)", "the pictures (FLUX)")
PLACE_PROMPT = "a close-up photograph of a weathered natural surface, richly detailed"
PICTURES_PROMPT = ("A close-up photograph of one continuous material surface with the texture, grain and colours "
                   "of the reference images, filling the frame.")


def _labels():
    from . import ip_labels
    return ip_labels


def drafts_dir() -> str:
    from . import paths as _paths
    return _paths.ensure_dir(os.path.join(_paths.data_dir(), "cache", "generate"))


def kept_dir() -> str:
    from . import paths as _paths
    return _paths.ensure_dir(os.path.join(_paths.subdir("output"), "generated", "materials",
                                          datetime.date.today().isoformat()))


# ─────────────────────────── sources ───────────────────────────

def source_label(src: dict):
    """The IP label a source stands for. A set's label is made from it the first time (from the index, in
    about a second); a picture becomes a one-picture label kept beside the others."""
    L = _labels()
    kind, ident = src.get("kind", "label"), str(src.get("id") or "").strip()
    if not ident:
        raise ValueError("A source needs an id: a label, a set or a picture.")
    if kind == "label":
        return L.load(ident)
    if kind == "set":
        slug = L.slugify(ident)
        try:
            return L.load(slug)
        except ValueError:
            from . import boards
            rec = boards.load(ident)
            if rec is None:
                raise ValueError(f"No set called `{ident}`.")
            paths, _ = boards.present(rec)
            members, files, world, stats = L.vectors_for(paths)
            return L.save(rec.get("name") or ident, members, files, {"kind": "set", "set": ident, **stats}, world,
                          slug=slug)
    if kind == "picture":
        slug = "picture-" + hashlib.md5(ident.encode("utf-8")).hexdigest()[:10]
        try:
            return L.load(slug)
        except ValueError:
            members, files, world, stats = L.vectors_for([ident])
            return L.save(os.path.basename(ident), members, files, {"kind": "picture", "picture": ident, **stats},
                          world, slug=slug)
    raise ValueError(f"A source is a label, a set or a picture, not {kind!r}.")


def resolved(sources: list[dict]):
    """[(label, weight)] with the weights as shares; sources with no weight are dropped."""
    out = [(source_label(s), max(0.0, float(s.get("weight", 1.0)))) for s in sources or []]
    out = [(lab, w) for lab, w in out if w > 0]
    total = sum(w for _, w in out)
    if not out or total <= 0:
        return []
    return [(lab, w / total) for lab, w in out]


def place(sources: list[dict], push: float = 2.0, spread: float = 0.0, seed: int = 0, axis: str | None = None,
          amount: float = 0.0) -> np.ndarray:
    """The blended position of the sources: each label's point (pushed from the average picture), mixed on
    the sphere by weight, at the length a real picture's vector has. `spread` slides toward one member of a
    source drawn by weight, so seeds give different looks of one place."""
    L = _labels()
    parts = resolved(sources)
    if not parts:
        raise ValueError("Choose at least one source: a set, a label or a picture.")
    vecs = [L.point(lab, push=push) for lab, _ in parts]
    weights = np.array([w for _, w in parts])
    units = np.stack([v / np.linalg.norm(v) for v in vecs])
    norm = float(sum(w * np.linalg.norm(v) for v, w in zip(vecs, weights)))
    v = (weights[:, None] * units).sum(0)
    v = v / (np.linalg.norm(v) + 1e-8) * norm
    if spread > 0:
        rng = np.random.default_rng(int(seed))
        lab = parts[int(rng.choice(len(parts), p=weights))][0]
        if len(lab.members):
            v = L.slerp(v, lab.members[int(rng.integers(len(lab.members)))], float(spread))
            v = v / np.linalg.norm(v) * norm
    return L.along(v.astype(np.float32), axis, amount)


def references(sources: list[dict], n: int = 4) -> list[str]:
    """Up to `n` photographs shared out among the sources by weight, each source's nearest its centre and
    unlike each other: what FLUX is shown."""
    parts = resolved(sources)
    if not parts:
        return []
    counts = [max(1, int(round(w * n))) for _, w in parts]
    while sum(counts) > n:
        counts[int(np.argmax(counts))] -= 1
    out = []
    for (lab, _), c in zip(parts, counts):
        out += lab.representatives(c) if c > 0 else []
    return out[:n]


# ─────────────────────────── drawing ───────────────────────────

@contextlib.contextmanager
def tiling(*modules):
    """Convolutions padded circularly while inside: what is drawn wraps at its edges, so it repeats without
    a seam at any size. Every convolution is put back as it was."""
    import torch

    changed = []
    for m in modules:
        for layer in m.modules():
            if isinstance(layer, torch.nn.Conv2d) and layer.padding_mode != "circular":
                changed.append(layer)
                layer.padding_mode = "circular"
    try:
        yield
    finally:
        for layer in changed:
            layer.padding_mode = "zeros"


def _open(path):
    from PIL import Image, ImageOps

    im = Image.open(path)
    im.draft("RGB", (1024, 1024))
    return ImageOps.exif_transpose(im).convert("RGB")


def draw_one(sources, words="", engine=ENGINES[0], seed=0, size=512, tileable=False, push=2.0, spread=0.25,
             vec=None, axis=None, amount=0.0):
    """One material. `vec` (a place already computed, for walks) stands in for the sources' place."""
    words = (words or "").strip()
    if engine == ENGINES[1]:
        from .alchemy import editor as ed

        refs = [_open(p) for p in references(sources)] if sources else []
        if not refs and not words:
            raise ValueError("Drawing from the pictures needs sources or words.")
        prompt = PICTURES_PROMPT if not words else (
            f"A close-up photograph of {words}, with the texture, grain and colours of the reference images, "
            f"filling the frame." if refs else f"A close-up photograph of {words}, richly detailed, filling the frame.")
        return ed.EDITOR.generate(prompt, refs, (size, size), steps=6, seed=seed)
    from . import style_transfer as st

    if vec is None:
        vec = place(sources, push=push, spread=spread, seed=seed, axis=axis, amount=amount) if sources else None
    if vec is None and not words:
        raise ValueError("Drawing from the place needs sources or words.")
    prompt = f"a close-up photograph of {words}, richly detailed" if words else PLACE_PROMPT
    scale = 0.0 if vec is None else (0.9 if not words else 0.7)
    v = np.zeros(1024, np.float32) if vec is None else vec
    if not tileable:
        return st.label_swatch(v, seed=seed, size=size, scale=scale, prompt=prompt)
    pipe = st._T2I_CACHE.get("t2i")              # built by a first, tiny draw; tiling patches its convolutions
    if pipe is None:
        st.label_swatch(v, seed=seed, size=64, steps=1, scale=scale, prompt=prompt)
        pipe = st._T2I_CACHE["t2i"]
    with tiling(pipe.unet, pipe.vae):
        return st.label_swatch(v, seed=seed, size=size, scale=scale, prompt=prompt)


def _draft(image, meta: dict) -> str:
    stamp = f"{time.strftime('%Y%m%d-%H%M%S')}-{hashlib.md5(json.dumps(meta, sort_keys=True, default=str).encode()).hexdigest()[:6]}"
    path = os.path.join(drafts_dir(), stamp + ".png")
    from PIL import PngImagePlugin

    info = PngImagePlugin.PngInfo()
    info.add_text("arcana:material", json.dumps(meta, default=str))
    image.save(path, "PNG", pnginfo=info)
    return path


def draw(sources, words="", engine=ENGINES[0], n=4, seed=0, size=512, tileable=False, push=2.0, spread=0.25,
         progress=None, cancelled=None, axis=None, amount=0.0) -> list[dict]:
    """`n` materials, one per seed from `seed`: [{"path", "seed"}], drafts until kept."""
    out = []
    for i in range(int(n)):
        if cancelled and cancelled():
            break
        s = int(seed) + i
        if progress:
            progress(i / max(1, n), f"drawing {i + 1} of {n}")
        im = draw_one(sources, words, engine, s, size, tileable, push, spread, axis=axis, amount=amount)
        meta = {"sources": sources, "words": words, "engine": engine, "seed": s, "size": size, "tileable": tileable,
                "push": push, "spread": spread, "axis": axis, "amount": amount}
        out.append({"path": _draft(im, meta), "seed": s, "meta": meta})
    return out


def walk(a, b, words="", frames=6, seed=0, size=512, tileable=False, push=2.0, progress=None, cancelled=None):
    """The great circle from the place of sources `a` to that of `b`, drawn at `frames` points with the seed
    held: the material turning into the other. Drawn from the place, the only engine that reads positions."""
    L = _labels()
    va, vb = place(a, push=push), place(b, push=push)
    out = []
    for i in range(int(frames)):
        if cancelled and cancelled():
            break
        t = i / max(1, frames - 1)
        if progress:
            progress(i / max(1, frames), f"walking {i + 1} of {frames}")
        v = L.slerp(va, vb, t)
        im = draw_one([], words, ENGINES[0], seed, size, tileable, vec=v)
        meta = {"walk": {"a": a, "b": b, "t": round(t, 3)}, "words": words, "seed": seed, "size": size,
                "tileable": tileable, "push": push}
        out.append({"path": _draft(im, meta), "seed": seed, "t": round(t, 3), "meta": meta})
    return out


def read_meta(path: str) -> dict:
    from PIL import Image

    with Image.open(path) as im:
        try:
            return json.loads(im.info.get("arcana:material") or "{}")
        except ValueError:
            return {}


def keep(draft: str) -> str:
    """A draft written to output/generated/materials/<date>/, its settings inside the PNG. The caller records it
    in the ledger as generated."""
    import shutil

    if not os.path.isfile(draft) or not os.path.abspath(draft).startswith(os.path.abspath(drafts_dir())):
        raise ValueError("That is not a drawn material.")
    meta = read_meta(draft)
    words = (meta.get("words") or "").strip()
    names = [str(s.get("id")) for s in meta.get("sources") or []] or (["walk"] if meta.get("walk") else [])
    stem = "-".join(x for x in ([words] if words else []) + names)[:60] or "material"
    stem = "".join(c if c.isalnum() or c in "-_" else "-" for c in stem.lower()).strip("-") or "material"
    dest = os.path.join(kept_dir(), f"{stem}__{time.strftime('%H%M%S')}.png")
    shutil.copy2(draft, dest)
    return dest

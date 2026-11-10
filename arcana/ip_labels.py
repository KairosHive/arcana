# ip_labels.py -- named positions in IP-Adapter space, measured on pictures
#
# An IP-Adapter steers Stable Diffusion 1.5 with one 1024-number vector: what
# its image encoder (OpenCLIP ViT-H/14) makes of a picture. Nothing about that
# vector has to come from one picture. An IP label is a place in that space
# measured on many: the centre of a set's vectors, the principal axes they
# spread along, and the members themselves. Alchemy draws from a label the way
# it draws from a material picture, and can mix two labels, move a material
# toward one, or push a label further from the average picture.
#
# Arcana's best search model is the same encoder: a dataset indexed with
# CLIP ViT-H/14 already holds the IP-Adapter vector of every picture (measured
# on 24 textures: cosine 0.999 to the adapter's encoder, the same length,
# about 21.6). A label made from such pictures reads the index and encodes
# nothing. Pictures from any other dataset, and crops, go through the adapter's
# own encoder.
#
# Ported from latent-lab's ip_space.py (build, point, crop_points, slerp). The
# lesson carried over: a mean is shorter than what it averages and the adapter
# reads a short vector as a faint one, so every point is rescaled to the length
# real pictures have; and the bare centre of many pictures is generic and greys
# out, which is what `spread` (a real member's vector, slerped toward) answers.
#
# A label is <data dir>/alchemy/labels/<slug>.npz:
#
#   members  [N, 1024]  the members' vectors, raw (not unit)
#   files    [N]        the picture each came from
#   centre   [1024]     their mean
#   axes     [A, 1024]  principal axes, A <= 16
#   spread   [A]        the standard deviation along each
#   norm     ()         the median length of a member
#   world    [1024]     the average picture of the datasets it came from, which
#                       `push` measures away from (zeros when unknown)
#   meta                JSON: name, origin, created

from __future__ import annotations

import json
import os
import re
import time

import numpy as np

DIM = 1024                         # OpenCLIP ViT-H/14, the SD 1.5 IP-Adapter's encoder
ENCODER = ("h94/IP-Adapter", "models/image_encoder")
CROP_DETAIL = 3.0                  # a crop sees a third of the frame's short side, as latent-lab trains
MAX_AXES = 16


# ─────────────────────────── storage ───────────────────────────

def labels_dir() -> str:
    try:
        from . import paths as _paths
    except ImportError:                               # pragma: no cover
        import paths as _paths
    return _paths.ensure_dir(os.path.join(_paths.data_dir(), "alchemy", "labels"))


def slugify(name: str) -> str:
    s = re.sub(r"[^\w\s-]", "", (name or "").strip().lower(), flags=re.UNICODE)
    return re.sub(r"[\s_-]+", "-", s).strip("-") or "label"


def _file(slug: str) -> str:
    if slug != slugify(slug):
        raise ValueError(f"`{slug}` cannot be a label id. Try `{slugify(slug)}`.")
    return os.path.join(labels_dir(), slug + ".npz")


class Label:
    """A label, read once and kept while its file is unchanged."""

    def __init__(self, slug: str, data: dict):
        self.slug = slug
        self.members = data["members"].astype(np.float32)
        self.files = [str(f) for f in data["files"]]
        self.centre = data["centre"].astype(np.float32)
        self.axes = data["axes"].astype(np.float32)
        self.spread = data["spread"].astype(np.float32)
        self.norm = float(data["norm"])
        self.world = data["world"].astype(np.float32)
        self.meta = json.loads(str(data["meta"]))

    @property
    def name(self) -> str:
        return self.meta.get("name", self.slug)

    def nearest(self, n: int = 6) -> list[str]:
        """The members closest to the centre: the pictures that show what the label is."""
        if not len(self.members):
            return []
        u = self.members / np.linalg.norm(self.members, axis=1, keepdims=True)
        c = self.centre / (np.linalg.norm(self.centre) + 1e-8)
        order = np.argsort(-(u @ c))
        seen, out = set(), []
        for i in order:
            if self.files[i] not in seen:
                seen.add(self.files[i])
                out.append(self.files[i])
            if len(out) >= n:
                break
        return out

    def representatives(self, n: int = 4, pool: int = 3) -> list[str]:
        """`n` pictures that show the label: chosen among the `pool` x n members nearest its centre, each as
        unlike the ones already chosen as possible, so a burst of near-identical frames gives one, not all."""
        if not len(self.members):
            return []
        u = self.members / np.linalg.norm(self.members, axis=1, keepdims=True)
        c = self.centre / (np.linalg.norm(self.centre) + 1e-8)
        order, seen = [], set()
        for i in np.argsort(-(u @ c)):
            if self.files[i] not in seen:
                seen.add(self.files[i])
                order.append(int(i))
            if len(order) >= n * pool:
                break
        chosen = [order[0]]
        while len(chosen) < min(n, len(order)):
            rest = [i for i in order if i not in chosen]
            far = max(rest, key=lambda i: min(1 - float(u[i] @ u[j]) for j in chosen))
            chosen.append(far)
        return [self.files[i] for i in chosen]

    def public(self) -> dict:
        return {"slug": self.slug, "name": self.name, "origin": self.meta.get("origin", {}),
                "members": int(len(self.members)), "pictures": len(set(self.files)),
                "created": self.meta.get("created", 0), "nearest": self.nearest(6),
                "tightness": tightness(self)}


_CACHE: dict = {}


def load(slug: str) -> Label:
    path = _file(slug)
    if not os.path.isfile(path):
        have = ", ".join(names()) or "none yet"
        raise ValueError(f"No IP label called `{slug}`. Labels: {have}.")
    stamp = os.path.getmtime(path)
    hit = _CACHE.get(slug)
    if hit and hit[0] == stamp:
        return hit[1]
    with np.load(path, allow_pickle=False) as z:
        lab = Label(slug, {k: z[k] for k in z.files})
    _CACHE[slug] = (stamp, lab)
    return lab


def names() -> list[str]:
    """Every label's id, newest first."""
    try:
        files = [f for f in os.listdir(labels_dir()) if f.endswith(".npz")]
    except OSError:
        return []
    files.sort(key=lambda f: os.path.getmtime(os.path.join(labels_dir(), f)), reverse=True)
    return [f[:-4] for f in files]


def options() -> tuple[str, ...]:
    return ("none",) + tuple(names())


def listing() -> list[dict]:
    out = []
    for slug in names():
        try:
            out.append(load(slug).public())
        except Exception as e:                        # an unreadable file is reported, not fatal
            out.append({"slug": slug, "name": slug, "error": str(e)})
    return out


def delete(slug: str) -> bool:
    path = _file(slug)
    if not os.path.isfile(path):
        return False
    os.remove(path)
    _CACHE.pop(slug, None)
    return True


def save(name: str, members: np.ndarray, files: list[str], origin: dict, world: np.ndarray | None = None,
         slug: str | None = None) -> Label:
    members = np.asarray(members, np.float32)
    if members.ndim != 2 or members.shape[1] != DIM or not len(members):
        raise ValueError(f"A label needs at least one {DIM}-number vector, got {members.shape}.")
    centre = members.mean(0)
    if len(members) > 1:
        _, s, vt = np.linalg.svd(members - centre, full_matrices=False)
        keep = min(MAX_AXES, len(s))
        axes, spread = vt[:keep], s[:keep] / np.sqrt(len(members) - 1)
    else:
        axes, spread = np.zeros((0, DIM), np.float32), np.zeros(0, np.float32)
    slug = slug or slugify(name)
    meta = {"name": name.strip() or slug, "origin": origin, "created": time.time()}
    path = _file(slug)
    tmp = path[:-4] + ".tmp.npz"
    np.savez(tmp, members=members, files=np.array(files), centre=centre.astype(np.float32),
             axes=axes.astype(np.float32), spread=spread.astype(np.float32),
             norm=np.float32(np.median(np.linalg.norm(members, axis=1))),
             world=(np.zeros(DIM, np.float32) if world is None else np.asarray(world, np.float32)),
             meta=np.array(json.dumps(meta)))
    os.replace(tmp, path)
    _CACHE.pop(slug, None)
    return load(slug)


# ─────────────────────────── geometry ───────────────────────────

def slerp(a, b, t: float) -> np.ndarray:
    """Along the sphere rather than through it, lengths interpolated: a straight line dips toward the
    mean and reads faint. t past 1 keeps going beyond b."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    ua, ub = a / max(na, 1e-8), b / max(nb, 1e-8)
    angle = np.arccos(np.clip(ua @ ub, -1.0, 1.0))
    if angle < 1e-4:
        return (a + t * (b - a)).astype(np.float32)
    u = (np.sin((1 - t) * angle) * ua + np.sin(t * angle) * ub) / np.sin(angle)
    return (u * ((1 - t) * na + t * nb)).astype(np.float32)


def tightness(lab: Label) -> float | None:
    """How alike the members are: mean cosine of a member to the centre. Near 1 one look, low a mixed bag."""
    if len(lab.members) < 2:
        return None
    u = lab.members / np.linalg.norm(lab.members, axis=1, keepdims=True)
    c = lab.centre / (np.linalg.norm(lab.centre) + 1e-8)
    return round(float((u @ c).mean()), 3)


def point(lab: Label, push: float = 2.0, spread: float = 0.0, seed: int = 0) -> np.ndarray:
    """A vector for the adapter from a label, at the length a real picture has.

    push   1 is the label's centre; higher moves it further from the average picture of its
           datasets (`world`), toward what sets the label apart; below 1 back toward the average.
    spread 0 is the centre; toward 1 a real member's vector, chosen by `seed`, which keeps what
           made that picture itself instead of the centre's generic grey.
    """
    v = lab.centre.astype(np.float64)
    if push != 1.0 and np.any(lab.world):
        v = lab.world + push * (v - lab.world)
    if spread > 0 and len(lab.members) > 1:
        m = lab.members[np.random.default_rng(int(seed)).integers(len(lab.members))]
        v = slerp(v, m, float(spread))
    v = np.asarray(v, np.float64)
    return (v * (lab.norm / max(np.linalg.norm(v), 1e-8))).astype(np.float32)


def steer(label: str | None, label2: str | None = None, mix: float = 0.5, push: float = 2.0,
          spread: float = 0.0, seed: int = 0, axis: str | None = None, amount: float = 0.0) -> np.ndarray | None:
    """The vector a method's label knobs ask for, or None when no label is chosen; moved along a concept
    axis by `amount` when one is given."""
    if not label or label == "none":
        return None
    v = point(load(label), push, spread, seed)
    if label2 and label2 != "none" and mix > 0:
        v = slerp(v, point(load(label2), push, spread, seed), float(mix))
    return along(v, axis, amount)


def toward(material_vec: np.ndarray, label_vec: np.ndarray, t: float) -> np.ndarray:
    """A material's vector moved toward a label: 0 the material, 1 the label, past 1 beyond it."""
    return slerp(material_vec, label_vec, float(t))


# ─────────────────────────── vectors for pictures ───────────────────────────

_ENC: dict = {}


def encoder():
    """The adapter's own image encoder, loaded once (half precision on a GPU)."""
    if "model" not in _ENC:
        import torch
        from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

        try:
            from . import gpu as _gpu
        except ImportError:                           # pragma: no cover
            import gpu as _gpu
        device = _gpu.device()
        dtype = torch.float16 if device == "cuda" else torch.float32
        model = CLIPVisionModelWithProjection.from_pretrained(ENCODER[0], subfolder=ENCODER[1], dtype=dtype)
        _ENC.update(model=model.to(device).eval(), proc=CLIPImageProcessor(), device=device, dtype=dtype)
    return _ENC


def release_encoder() -> None:
    """Free the encoder's memory, for a LoRA run that wants the whole card."""
    if _ENC:
        _ENC.clear()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass


def encode(images) -> np.ndarray:
    """[N, 1024] adapter vectors for PIL images, each read as the adapter reads one (224, centre)."""
    import torch

    enc = encoder()
    out = []
    for start in range(0, len(images), 16):
        px = enc["proc"](images=images[start:start + 16], return_tensors="pt").pixel_values
        with torch.no_grad():
            v = enc["model"](pixel_values=px.to(enc["device"], enc["dtype"])).image_embeds
        out.append(v.float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, DIM), np.float32)


def crops(path: str, count: int, detail: float = CROP_DETAIL, seed: int = 0):
    """`count` square crops of a picture, each seeing 1/detail of its short side: the grain a whole
    picture squeezed into 224 pixels loses."""
    from PIL import Image, ImageOps

    rng = np.random.default_rng(seed)
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
    side = max(32, int(min(im.size) / detail))
    out = []
    for _ in range(count):
        x = int(rng.integers(0, im.width - side + 1))
        y = int(rng.integers(0, im.height - side + 1))
        out.append(im.crop((x, y, x + side, y + side)).resize((224, 224), Image.LANCZOS))
    return out


def _vith_datasets() -> list[str]:
    """The image datasets indexed with CLIP ViT-H/14 (1024 numbers): their vectors are the adapter's."""
    try:
        from . import core
    except ImportError:                               # pragma: no cover
        import core
    out = []
    for d in core.find_datasets():
        if d.get("modality", "image") != "image":
            continue
        try:
            ds = core.open_dataset(d["name"])
        except Exception:
            continue
        if int(ds.index.ndim) == DIM:
            out.append(d["name"])
    return out


_REVERSE: dict = {}
_WORLD: dict = {}


def _reverse(name: str):
    try:
        from . import core
    except ImportError:                               # pragma: no cover
        import core
    ds = core.open_dataset(name)
    hit = _REVERSE.get(name)
    if hit is None or hit[0] is not ds.idx2path:
        hit = (ds.idx2path, {os.path.normcase(os.path.abspath(p)): k for k, p in ds.idx2path.items()}, ds)
        _REVERSE[name] = hit
    return hit[1], hit[2]


def world(name: str) -> np.ndarray:
    """The average picture of a ViT-H dataset, read from its index."""
    rev, ds = _reverse(name)
    hit = _WORLD.get(name)
    if hit is None or hit[0] != len(rev):
        keys = list(ds.idx2path)
        v = np.stack([ds.index.get(k) for k in keys]).astype(np.float32).mean(0)
        hit = (len(rev), v)
        _WORLD[name] = hit
    return hit[1]


def vectors_for(paths: list[str], per_picture: int = 0, progress=None, cancelled=None):
    """(members, files, world, stats) for pictures.

    With per_picture 0 each picture gives one vector: read from a ViT-H index when one holds the
    picture, encoded otherwise. With per_picture > 0 each gives that many crops at CROP_DETAIL, all
    encoded, which keeps the grain and takes seconds per hundred pictures on a GPU.
    """
    from PIL import Image, ImageOps

    members, files, from_index, used = [], [], 0, set()
    sources = _vith_datasets() if per_picture == 0 else []
    reverse = {name: _reverse(name) for name in sources}
    to_encode = []
    for p in paths:
        norm = os.path.normcase(os.path.abspath(p))
        hit = None
        for name, (rev, ds) in reverse.items():
            k = rev.get(norm)
            if k is not None:
                hit = (name, ds.index.get(k))
                break
        if hit is not None:
            members.append(np.asarray(hit[1], np.float32))
            files.append(p)
            used.add(hit[0])
            from_index += 1
        else:
            to_encode.append(p)
    batch, owners = [], []
    for i, p in enumerate(to_encode):
        if cancelled and cancelled():
            raise RuntimeError("Stopped.")
        try:
            if per_picture:
                ims = crops(p, per_picture, seed=i)
            else:
                with Image.open(p) as im:
                    ims = [ImageOps.exif_transpose(im).convert("RGB")]
        except OSError:
            continue
        batch += ims
        owners += [p] * len(ims)
        if len(batch) >= 32 or i == len(to_encode) - 1:
            members += list(encode(batch))
            files += owners
            batch, owners = [], []
        if progress:
            progress((i + 1) / max(1, len(to_encode)), f"encoded {i + 1} of {len(to_encode)} pictures")
    if batch:
        members += list(encode(batch))
        files += owners
    if not members:
        raise ValueError("None of these pictures could be read.")
    world_from = sorted(used) or _vith_datasets()
    w = np.mean([world(n) for n in world_from], 0) if world_from else None
    stats = {"from_index": from_index, "encoded": len(to_encode), "worlds": world_from}
    return np.stack(members), files, w, stats


def search_paths(text: str, dataset: str, n: int) -> list[str]:
    """The `n` pictures of a dataset that best match words, by the dataset's own CLIP."""
    try:
        from . import core
    except ImportError:                               # pragma: no cover
        import core
    ds = core.open_dataset(dataset)
    return [p for _, p, _ in core.search(ds.index, ds.idx2path, text, n)]


def nearest_in(dataset: str, vec: np.ndarray, n: int, allowed=None):
    """The pictures of a ViT-H dataset nearest a label's vector: a label used as a search."""
    try:
        from . import core
    except ImportError:                               # pragma: no cover
        import core
    ds = core.open_dataset(dataset)
    if int(ds.index.ndim) != DIM:
        raise ValueError(f"{dataset} was indexed with a {ds.index.ndim}-number CLIP, and IP labels live in "
                         f"CLIP ViT-H/14's 1024. Search with a label in a dataset indexed with ViT-H/14.")
    return core.search(ds.index, ds.idx2path, np.asarray(vec, np.float32), n, allowed=allowed)


# ─────────────────────────── concept axes ───────────────────────────
#
# A concept is a direction in IP space, from one place to another: the difference of two places'
# means, as latent-lab's concepts.py fitted `humid` (wet against dry). Each end is a set, a label or
# the matches for some words. The same direction sorts the archive (each picture's projection on it)
# and moves a place along it, so the ruler and the knob are one object. An amount of 1 moves a place by
# the distance between the two ends (the whole way from dry to wet); the spread within an end is far
# smaller (the first axis fitted, wet, has its ends 94 within-spreads apart), so it makes a poor unit.

def axes_dir() -> str:
    try:
        from . import paths as _paths
    except ImportError:                               # pragma: no cover
        import paths as _paths
    return _paths.ensure_dir(os.path.join(_paths.data_dir(), "alchemy", "axes"))


class Axis:
    def __init__(self, slug: str, data: dict):
        self.slug = slug
        self.direction = data["direction"].astype(np.float32)
        self.spread = float(data["spread"])
        self.meta = json.loads(str(data["meta"]))

    @property
    def name(self) -> str:
        return self.meta.get("name", self.slug)

    def public(self) -> dict:
        return {"slug": self.slug, "name": self.name, "pos": self.meta.get("pos"), "neg": self.meta.get("neg"),
                "apart": self.meta.get("apart"), "created": self.meta.get("created", 0)}


def _axis_file(slug: str) -> str:
    if slug != slugify(slug):
        raise ValueError(f"`{slug}` cannot be an axis id. Try `{slugify(slug)}`.")
    return os.path.join(axes_dir(), slug + ".npz")


def axis_names() -> list[str]:
    try:
        files = [f for f in os.listdir(axes_dir()) if f.endswith(".npz")]
    except OSError:
        return []
    files.sort(key=lambda f: os.path.getmtime(os.path.join(axes_dir(), f)), reverse=True)
    return [f[:-4] for f in files]


def axis_options() -> tuple[str, ...]:
    return ("none",) + tuple(axis_names())


def load_axis(slug: str) -> Axis:
    path = _axis_file(slug)
    if not os.path.isfile(path):
        raise ValueError(f"No axis called `{slug}`. Axes: {', '.join(axis_names()) or 'none yet'}.")
    with np.load(path, allow_pickle=False) as z:
        return Axis(slug, {k: z[k] for k in z.files})


def axis_listing() -> list[dict]:
    return [load_axis(s).public() for s in axis_names()]


def delete_axis(slug: str) -> bool:
    path = _axis_file(slug)
    if not os.path.isfile(path):
        return False
    os.remove(path)
    return True


def make_axis(name: str, pos: np.ndarray, neg: np.ndarray, origin: dict, slug: str | None = None) -> Axis:
    """An axis from two groups of member vectors: from the mean of `neg`'s unit vectors to the mean of
    `pos`'s. `spread` (the knob's unit) is the distance between the two ends along it; `apart` is that
    distance in units of how far a member strays within its end (near 0: this encoder cannot tell them
    apart)."""
    up = pos / np.linalg.norm(pos, axis=1, keepdims=True)
    un = neg / np.linalg.norm(neg, axis=1, keepdims=True)
    d = up.mean(0) - un.mean(0)
    d = d / (np.linalg.norm(d) + 1e-8)
    proj = np.concatenate([up @ d, un @ d])
    within = float(np.sqrt(((up @ d - (up @ d).mean()) ** 2).sum() + ((un @ d - (un @ d).mean()) ** 2).sum())
                   / max(1, len(proj) - 2))
    between = float((up @ d).mean() - (un @ d).mean())
    spread = max(between, 1e-4)
    apart = float(between / max(within, 1e-6))
    slug = slug or slugify(name)
    meta = {"name": name.strip() or slug, **origin, "apart": round(apart, 2), "created": time.time()}
    path = _axis_file(slug)
    tmp = path[:-4] + ".tmp.npz"
    np.savez(tmp, direction=d.astype(np.float32), spread=np.float32(spread), meta=np.array(json.dumps(meta)))
    os.replace(tmp, path)
    return load_axis(slug)


def along(v: np.ndarray, axis: str | None, amount: float) -> np.ndarray:
    """A vector moved `amount` along an axis (1 is the distance between its two ends), at the length it had."""
    if not axis or axis == "none" or not amount:
        return v
    ax = load_axis(axis)
    n = np.linalg.norm(v)
    u = v / max(n, 1e-8) + float(amount) * ax.spread * ax.direction
    return (u / max(np.linalg.norm(u), 1e-8) * n).astype(np.float32)


def rank_along(dataset: str, axis: str, n: int = 40, allowed=None):
    """The pictures of a ViT-H dataset furthest along an axis, [(key, path, projection)], highest first."""
    try:
        from . import core
    except ImportError:                               # pragma: no cover
        import core
    ds = core.open_dataset(dataset)
    if int(ds.index.ndim) != DIM:
        raise ValueError(f"{dataset} was indexed with a {ds.index.ndim}-number CLIP; axes live in ViT-H/14's {DIM}.")
    ax = load_axis(axis)
    keys = [k for k in ds.idx2path if allowed is None or k in allowed]
    vecs = np.stack([ds.index.get(k) for k in keys]).astype(np.float32)
    proj = (vecs / np.linalg.norm(vecs, axis=1, keepdims=True)) @ ax.direction
    order = np.argsort(-proj)[:n]
    return [(keys[i], ds.idx2path[keys[i]], float(proj[i])) for i in order]

"""Alchemy in token space: pictures as grids of tokens, mixed by arithmetic and drawn back.

A picture and its materials become grids of tokens, one per 16 x 16 pixels, in one of two spaces
(spaces.py). Three methods work on those grids:

- mix    the materials laid into the picture: WHERE each goes (regions along the picture's light,
         depth or a side; blended at every place; or each place taking the material it resembles),
         WHAT each material carries (its tokens, its statistics, its whole distribution by optimal
         transport, its principal axes, or a push along its mean direction), and the OPERATION that
         joins picture and material tokens (a straight mix, an arc, a sum, a difference, a product,
         the larger or smaller channel, or a frequency split).
- mosh   tokens moved out of their places: rebuilt from the closest matches (in the FLUX.2 VAE,
         DINO decides which piece goes where and the VAE draws it), dragged, or sorted.
- weave  new textures from the materials alone: blended, patchwork, one's layout in another's
         grain, or remixed.

Every mix is put back where real tokens lie before it is drawn: on RAE's shell, or with each
channel's spread restored in the VAE. Ported from latent-lab's registry, where each piece was
measured on photographs.
"""

from __future__ import annotations

import collections
import threading

import numpy as np

from . import mask as mk
from . import spaces

LUM = mk.LUM
TIP = 0.15   # how far a weight tips a rebuilt piece's choice, in cosine: half the weight costs 0.1

MIXES = {"regions": "each material in its own part of the selection, along the picture's light, depth or a side",
         "blend": "every material at every place, mixed token by token by the weights",
         "nearest": "each place takes the material it already resembles; the weights are the shares"}
ORDERS = {"light": "the first material in the darkest parts, the last in the lightest",
          "depth": "the first material on what is nearest, the last on what is furthest",
          "across": "the first material on one side of the frame, the last on the other"}
CARRIES = {"tokens": "the material's own tokens: its drawing replaces the picture's",
           "statistics": "the picture's tokens given the material's channel means and spreads",
           "direction": "the picture's tokens pushed along the material's mean direction",
           "transport": "the picture's tokens given the material's whole distribution (optimal transport)",
           "tone axes": "the material's first four principal axes (in the VAE: its tonality)",
           "detail axes": "the material's principal axes 5 to 24: its finer structure",
           "surface axes": "the material's principal axes 1 to 24: its whole surface on the picture's forms"}
AXES_SPANS = {"tone axes": (0, 4), "detail axes": (4, 24), "surface axes": (0, 24)}
OPERATIONS = {"replace": "a straight mix toward the material",
              "slerp": "along the arc between the tokens, keeping their length",
              "add": "the material added to the picture: both at once",
              "subtract": "the material taken away from the picture",
              "multiply": "channel by channel product: only what both carry stays strong",
              "max": "each channel the larger of the two",
              "min": "each channel the smaller of the two",
              "difference": "each channel the distance between the two: where they disagree",
              "split": "the picture's broad forms with the material's fine detail"}
MOSHES = {"rebuild": "the picture rebuilt from the closest pieces of the materials (or of itself)",
          "drag": "tokens pulled along one direction, each as far as a field says",
          "sort": "runs of the brightest places with their tokens sorted, a token at a time"}
WEAVES = {"blend": "the materials mixed token by token, walking from the first to the last",
          "patchwork": "patches of each material in a grown pattern, the decoder drawing every seam",
          "layout": "the first material's arrangement rebuilt from the others' tokens",
          "remix": "the first material's own pieces scattered over the frame"}
LOOKS = ("crispness", "warmth", "exposure", "contrast")

_cache: collections.OrderedDict = collections.OrderedDict()
_cache_lock = threading.Lock()


def column(a, like):
    import torch

    return torch.from_numpy(np.ascontiguousarray(np.asarray(a).reshape(-1, 1), np.float32)).to(like.device)


def encode(sp, space_name, key, image, grid):
    """A picture's token grid [rows*cols, D] as floats, kept for the last few dozen asked."""
    k = (space_name, key, grid)
    with _cache_lock:
        if k in _cache:
            _cache.move_to_end(k)
            return _cache[k]
    z = sp.encode(image, grid)[0].float()
    with _cache_lock:
        _cache[k] = z
        while len(_cache) > 40:
            _cache.popitem(last=False)
    return z


def decode(sp, z, grid):
    """Tokens drawn back to a picture [H, W, 3] in 0..1."""
    return sp.decode(sp.shell(z.reshape(1, grid[0] * grid[1], -1)), grid)[0]


def pool(a, rows, cols):
    """A field in the frame as one number per token, softened by about a token."""
    H, W = a.shape
    return mk.blur(a.reshape(rows, H // rows, cols, W // cols).mean((1, 3)), 0.7)


# ─────────────────────────── where ───────────────────────────

def weights_of(w, n):
    w = np.array([max(float(x), 0.0) for x in (list(w) + [1.0] * n)[:n]], np.float64)
    return w / w.sum() if w.sum() > 0 else np.full(n, 1.0 / n)


def order_of(how, c, side, inside, depth_fn):
    H, W = c.shape[:2]
    if how == "depth":
        f = 1.0 - depth_fn()
    elif how == "across":
        y, x = np.mgrid[0:H, 0:W].astype(np.float32)
        f = (y - H / 2) * np.cos(side) - (x - W / 2) * np.sin(side)
    else:
        f = mk.blur(c @ LUM, max(W, H) / 24)
    ref = f[inside] if inside is not None and inside.sum() > 64 else f.ravel()
    if ref.size > 20_000:
        ref = ref[np.random.default_rng(0).choice(ref.size, 20_000, replace=False)]
    ref = np.sort(ref)
    return np.interp(f, ref, np.linspace(0.0, 1.0, ref.size)).astype(np.float32)


def regions(f, w, seam):
    count = len(w)
    out = np.zeros((count,) + f.shape, np.float32)
    if count == 1:
        out[0] = 1
        return out
    edges = np.cumsum(w)[:-1]
    s = [mk.smoothstep(np.clip((f - edges[j]) / max(seam * min(w[j], w[j + 1]), 1e-3) + 0.5, 0, 1))
         for j in range(count - 1)]
    for k in range(count):
        out[k] = (s[k - 1] if k else 1.0) * (1.0 - (s[k] if k < count - 1 else 0.0))
    return out / np.maximum(out.sum(0), 1e-6)


def nearest(z_pic, raws, w, contrast, inside=None):
    import torch
    import torch.nn.functional as F

    means = F.normalize(torch.stack([r.mean(0) for r in raws]), dim=1)
    sim = F.normalize(z_pic, dim=1) @ means.T
    sim = 8.0 * float(contrast) * (sim - sim.mean(0)) / (sim.std(0) + 1e-6)
    want = torch.as_tensor(w, dtype=sim.dtype, device=sim.device)
    where = torch.ones(len(sim), 1, dtype=sim.dtype, device=sim.device) if inside is None else \
        torch.as_tensor(np.asarray(inside, np.float32).reshape(-1, 1), device=sim.device, dtype=sim.dtype)
    bias = torch.log(want + 1e-9)
    for _ in range(40):
        got = (torch.softmax(sim + bias, dim=1) * where).sum(0) / where.sum().clamp(min=1e-6)
        bias = bias + torch.log((want + 1e-9) / (got + 1e-9))
    return torch.softmax(sim + bias, dim=1).T.cpu().numpy()


def lineup(f, count, balance, seam):
    w = np.zeros((count,) + f.shape, np.float32)
    if count == 1:
        w[0] = 1
        return w
    pos = np.clip(f, 0, 1) ** (2.0 ** ((0.5 - balance) * 4)) * (count - 1)
    i = np.minimum(np.floor(pos), count - 2).astype(int)
    s = mk.smoothstep(np.clip((pos - i - 0.5) / max(seam, 1e-3) + 0.5, 0, 1))
    for k in range(count):
        w[k] = np.where(i == k, 1 - s, 0) + np.where(i == k - 1, s, 0)
    return w


# ─────────────────────────── what and how ───────────────────────────

def token_blur(z, grid, sigma):
    import torch
    import torch.nn.functional as F

    rows, cols = grid
    k = max(3, int(sigma * 3) * 2 + 1)
    x = torch.arange(k, device=z.device, dtype=z.dtype) - k // 2
    g = torch.exp(-x ** 2 / (2 * float(sigma) ** 2))
    g = g / g.sum()
    a = z.T.reshape(1, -1, rows, cols)
    d = a.shape[1]
    a = F.conv2d(F.pad(a, (k // 2, k // 2, 0, 0), mode="reflect"), g.view(1, 1, 1, k).repeat(d, 1, 1, 1), groups=d)
    a = F.conv2d(F.pad(a, (0, 0, k // 2, k // 2), mode="reflect"), g.view(1, 1, k, 1).repeat(d, 1, 1, 1), groups=d)
    return a.reshape(d, -1).T


def combined(p, m, t, how, grid=None, size=1.5):
    import torch
    import torch.nn.functional as F

    if how == "split":
        target = token_blur(p, grid, size) + m - token_blur(m, grid, size)
        return p + t * (target - p)
    if how == "add":
        return p + t * m
    if how == "subtract":
        return p - t * m
    if how == "slerp":
        a, b = F.normalize(p, dim=1), F.normalize(m, dim=1)
        omega = torch.acos((a * b).sum(1, keepdim=True).clamp(-0.9999, 0.9999))
        s = torch.sin(omega)
        arc = (torch.sin((1 - t) * omega) * a + torch.sin(t * omega) * b) / s
        return arc * ((1 - t) * p.norm(dim=1, keepdim=True) + t * m.norm(dim=1, keepdim=True))
    if how == "multiply":
        # About 60% of a token is the channel pattern every token of a grid shares; multiplied whole,
        # that pattern squared swamps the rest and every token comes out alike (one flat colour).
        dp, dm = p - p.mean(0), m - m.mean(0)
        prod = dp * dm
        target = p.mean(0) + prod / (prod.std() + 1e-6) * dp.std()
    else:
        target = {"max": lambda: torch.maximum(p, m), "min": lambda: torch.minimum(p, m),
                  "difference": lambda: (p - m).abs()}.get(how, lambda: m)()
    return p + t * (target - p)


def settled(sp, z, sources, weights):
    return sp.settle(z, sources, weights) if hasattr(sp, "settle") else z


def transported(p, m, steps=64, seed=0):
    import torch

    g = torch.Generator(device=p.device).manual_seed(seed)
    z = p.clone()
    idx = torch.linspace(0, len(m) - 1, len(p), device=p.device).round().long()
    for _ in range(steps):
        q, _ = torch.linalg.qr(torch.randn(p.shape[1], p.shape[1], generator=g, device=p.device))
        a = z @ q
        target = torch.empty_like(a)
        target.scatter_(0, a.argsort(0), (m @ q).sort(0).values[idx])
        z = z + (target - a) @ q.T
    return z


def carried(z_mat, z_pic, how, push=1.0, axes=None):
    if how == "tokens":
        return z_mat
    if how == "statistics":
        target = (z_pic - z_pic.mean(0)) / (z_pic.std(0) + 1e-6) * z_mat.std(0) + z_mat.mean(0)
    elif how == "direction":
        target = z_pic + (z_mat.mean(0) - z_pic.mean(0))
    elif how == "transport":
        target = transported(z_pic, z_mat)
    elif how in AXES_SPANS:
        mu, basis, _ = axes
        lo, hi = AXES_SPANS[how]
        b = basis.to(z_pic.dtype)
        cp, cm = (z_pic - mu) @ b.T, (z_mat - mu) @ b.T
        cp[:, lo:hi] = cm[:, lo:hi]
        target = cp @ b + mu
    else:
        return z_mat
    return z_pic + float(push) * (target - z_pic)


def looked(space_name, sp, z, k):
    """The tokens moved along the space's crispness, warmth, exposure and contrast axes."""
    if not any(float(k.get(name, 0) or 0) for name in LOOKS):
        return z
    (mu, basis, spread), ctl = spaces.axes(space_name, sp)
    for name, (i, sign) in ctl.items():
        v = float(k.get(name, 0) or 0)
        if v:
            z = z + (v * sign * spread[i]) * basis[i].to(z.dtype)
    return z


# ─────────────────────────── moving pieces ───────────────────────────

def pieces(grid, b, step):
    import torch.nn.functional as F

    R, C, D = grid.shape
    g = grid.permute(2, 0, 1)
    if step > 1:
        g = F.pad(g[None], (0, -C % step, 0, -R % step), mode="replicate")[0]
    u = g.unfold(1, b, step).unfold(2, b, step)
    ny, nx = u.shape[1:3]
    return u.permute(1, 2, 0, 3, 4).reshape(ny * nx, D, b, b), ny, nx


def laid(blocks, ny, nx, R, C):
    D, b = blocks.shape[1], blocks.shape[2]
    return blocks.reshape(ny, nx, D, b, b).permute(2, 0, 3, 1, 4).reshape(D, ny * b, nx * b)[:, :R, :C].permute(1, 2, 0)


def rebuilt(target, source, b, pick, same, gen, bias=None, draw=None):
    import torch
    import torch.nn.functional as F

    R, C, D = target.shape
    t, ny, nx = pieces(target, b, b)
    s, sy, sx = pieces(source, b, 1)
    sim = F.normalize(t, dim=1).flatten(1) @ F.normalize(s, dim=1).flatten(1).T
    if same:
        ty, tx = torch.meshgrid(torch.arange(ny) * b, torch.arange(nx) * b, indexing="ij")
        cy, cx_ = torch.meshgrid(torch.arange(sy), torch.arange(sx), indexing="ij")
        near = ((ty.reshape(-1, 1) - cy.reshape(1, -1)).abs() < b) & ((tx.reshape(-1, 1) - cx_.reshape(1, -1)).abs() < b)
        sim = sim.masked_fill(near.to(sim.device), -9.0)
    if bias is not None:
        sim = sim + bias(sy, sx).to(sim.device, sim.dtype)[None]
    k = max(1, min(int(pick), sim.shape[1]))
    top = sim.topk(k, dim=1).indices
    choice = torch.randint(0, k, (len(top),), generator=gen).to(top.device)
    if draw is not None:
        s = pieces(draw.to(s.device), b, 1)[0]
    return laid(s[top[torch.arange(len(top), device=top.device), choice]], ny, nx, R, C)


def scattered(grid, b, reach, gen):
    import torch

    R, C, D = grid.shape
    s, sy, sx = pieces(grid, b, 1)
    ny, nx = -(-R // b), -(-C // b)
    y = torch.arange(ny).repeat_interleave(nx) * b
    x = torch.arange(nx).repeat(ny) * b
    r = int(round(float(reach)))
    dy = torch.randint(-r, r + 1, (ny * nx,), generator=gen)
    dx = torch.randint(-r, r + 1, (ny * nx,), generator=gen)
    py = (y + dy).clamp(0, sy - 1)
    px = (x + dx).clamp(0, sx - 1)
    return laid(s[(py * sx + px).to(s.device)], ny, nx, R, C)


def dragged(source, field, length, angle):
    import torch

    R, C, D = source.shape
    y, x = torch.meshgrid(torch.arange(R, dtype=torch.float32), torch.arange(C, dtype=torch.float32), indexing="ij")
    f = torch.from_numpy(np.asarray(field, np.float32)) * float(length)
    sy = (y + f * np.cos(angle)).round().clamp(0, R - 1).long()
    sx = (x - f * np.sin(angle)).round().clamp(0, C - 1).long()
    return source[sy.to(source.device), sx.to(source.device)]


def sorted_runs(source, key, span, columns):
    import torch

    g, k = (source.transpose(0, 1), key.T) if columns else (source, key)
    out = g.clone()
    on = mk.rank(k) > 1.0 - float(span)
    for i in range(k.shape[0]):
        j = 0
        while j < k.shape[1]:
            if not on[i, j]:
                j += 1
                continue
            e = j
            while e < k.shape[1] and on[i, e]:
                e += 1
            order = torch.from_numpy(np.argsort(k[i, j:e], kind="stable") + j).to(g.device)
            out[i, j:e] = g[i, order]
            j = e
    return out.transpose(0, 1) if columns else out


# ─────────────────────────── the three methods ───────────────────────────
# Each takes the step (pipeline.Step) and returns the picture [H, W, 3] in 0..1 at the step's frame.

def _grid(step):
    return step.H // 16, step.W // 16


def mix(step, k):
    import torch  # noqa: F401

    name = k.get("model", "DINOv2")
    sp = spaces.space(name)
    rows, cols = _grid(step)
    mats = step.materials
    if not mats:
        raise ValueError("mix needs at least one material.")
    w = weights_of(step.weights, len(mats))
    how = k.get("where", "regions") if len(mats) > 1 else "blend"
    m = step.mask
    with spaces.lock():
        z_pic = encode(sp, name, step.picture_key, step.c, (rows, cols))
        raws = [encode(sp, name, mk_, t, (rows, cols)) for mk_, t in zip(step.material_keys, mats)]
        if how == "regions":
            f = order_of(k.get("order", "light"), step.c, float(k.get("side", 0.0)),
                         None if m is None else m > 0.5, step.depth)
            tok = np.stack([pool(x, rows, cols) for x in regions(f, w, float(k.get("seam", 0.35)))])
        elif how == "nearest":
            inside = None if m is None else pool(m, rows, cols)
            tok = nearest(z_pic, raws, w, float(k.get("firmness", 0.6)), inside).reshape(len(mats), rows, cols)
        else:
            tok = np.repeat(w.astype(np.float32)[:, None, None], rows * cols, 1).reshape(len(mats), rows, cols)
        if m is not None:
            tok = tok * pool(m, rows, cols)[None]
        share = tok.sum(0)
        carry = k.get("carry", "tokens")
        ax = spaces.axes(name, sp)[0] if carry in AXES_SPANS else None
        brought = [carried(r, z_pic, carry, float(k.get("push", 1.0)), ax) for r in raws]
        mat = sum(b * column(t, z_pic) for b, t in zip(brought, tok))
        mat = mat / column(np.maximum(share, 1e-6), z_pic)
        z = combined(z_pic, mat, column(share, z_pic), k.get("operation", "replace"), (rows, cols),
                     float(k.get("split_size", 1.5)))
        s = float(share.mean())
        z = settled(sp, z, [z_pic] + brought, [1 - s] + [s * float(t.mean()) / max(s, 1e-6) for t in tok])
        u = float(k.get("picture", 0.0))
        z = looked(name, sp, (1 - u) * z + u * z_pic, k)
        return decode(sp, z, (rows, cols))


def mosh(step, k):
    import torch

    name = k.get("model", "DINOv2")
    sp = spaces.space(name)
    rows, cols = _grid(step)
    way = k.get("way", "rebuild")
    from_materials = way == "rebuild" and bool(step.materials) and k.get("source", "materials") == "materials"
    gen = torch.Generator().manual_seed(int(k.get("seed", 0)))
    grid = lambda z: z.reshape(rows, cols, -1)  # noqa: E731
    with spaces.lock():
        z_pic = encode(sp, name, step.picture_key, step.c, (rows, cols))
        vae = hasattr(sp, "settle")
        seen_sp, seen_name = (spaces.space("DINOv2"), "DINOv2") if vae and way == "rebuild" else (sp, name)

        def seen(key, img):
            return encode(seen_sp, seen_name, key, img, (rows, cols))

        if way == "rebuild" and from_materials:
            w = weights_of(step.weights, len(step.materials))
            tip = torch.as_tensor([TIP * np.log(x / w.max()) if x > 0 else -9.0 for x in w], dtype=torch.float32)
            wide = torch.cat([grid(seen(kk, t)) for kk, t in zip(step.material_keys, step.materials)], 1)
            draw = torch.cat([grid(encode(sp, name, kk, t, (rows, cols)))
                              for kk, t in zip(step.material_keys, step.materials)], 1) if vae else None
            n = len(step.materials)
            owner = lambda sy, sx: tip[(torch.arange(sx) // cols).clamp(max=n - 1)].repeat(sy)  # noqa: E731
            moved = rebuilt(grid(seen(step.picture_key, step.c)), wide, int(k.get("pieces", 1)), int(k.get("wildness", 1)),
                            False, gen, owner, draw)
        elif way == "rebuild":
            moved = rebuilt(grid(seen(step.picture_key, step.c)), grid(seen(step.picture_key, step.c)),
                            int(k.get("pieces", 2)), int(k.get("wildness", 4)), True, gen,
                            draw=grid(z_pic) if vae else None)
        elif way == "drag":
            by = k.get("by", "light")
            small = lambda a: a.reshape(rows, a.shape[0] // rows, cols, a.shape[1] // cols).mean((1, 3))  # noqa: E731
            if by == "depth":
                field = mk.rank(small(step.depth()))
            elif by == "noise":
                field = mk.rank(mk.growth(rows, cols, 3.0, int(k.get("seed", 0))))
            else:
                field = mk.rank(small(step.c @ LUM))
            moved = dragged(grid(z_pic), field, float(k.get("length", 8.0)), float(k.get("side", np.pi)))
        else:
            lum = step.c.reshape(rows, 16, cols, 16, 3).mean((1, 3)) @ LUM
            moved = sorted_runs(grid(z_pic), lum, float(k.get("span", 0.5)), k.get("along", "columns") == "columns")
        moved = moved.reshape(rows * cols, -1)
        h = float(k.get("hold", 0.0))
        z = combined(z_pic, moved, column(np.full(rows * cols, 1.0 - h, np.float32), z_pic),
                     k.get("operation", "replace"), (rows, cols), float(k.get("split_size", 1.5)))
        z = settled(sp, z, [z_pic, moved], [h, 1.0 - h])
        if step.mask is not None:
            s = column(pool(step.mask, rows, cols), z)
            z = s * z + (1 - s) * z_pic
        z = looked(name, sp, z, k)
        return decode(sp, z, (rows, cols))


def weave(step, k):
    import torch

    name = k.get("model", "DINOv2")
    sp = spaces.space(name)
    rows, cols = _grid(step)
    mats = step.materials
    if not mats:
        raise ValueError("weave needs at least one material.")
    pattern = k.get("pattern", "blend")
    gen = torch.Generator().manual_seed(int(k.get("seed", 0)))
    with spaces.lock():
        grids = [encode(sp, name, kk, t, (rows, cols)) for kk, t in zip(step.material_keys, mats)]
        if pattern in ("blend", "patchwork"):
            if pattern == "blend":
                f = np.full((rows, cols), float(k.get("walk", 0.5)), np.float32)
                w = lineup(f, len(grids), 0.5, 1.0)
            else:
                f = mk.rank(mk.growth(rows, cols, float(k.get("patch", 8.0)) / 4, int(k.get("seed", 0))))
                w = lineup(f, len(grids), float(k.get("balance", 0.5)), float(k.get("seam", 0.35)))
                w = np.stack([mk.blur(x, 0.7) for x in w])
            z = sum(g * column(x, g) for g, x in zip(grids, w))
            z = settled(sp, z, grids, [float(x.mean()) for x in w])
        else:
            first = grids[0].reshape(rows, cols, -1)
            if pattern == "layout":
                others = [g.reshape(rows, cols, -1) for g in grids[1:]] or [first]
                moved = rebuilt(first, torch.cat(others, 1), int(k.get("pieces", 2)), int(k.get("wildness", 1)),
                                len(grids) == 1, gen)
            else:
                moved = scattered(first, int(k.get("pieces", 2)), float(k.get("reach", 12.0)), gen)
            h = float(k.get("hold", 0.0))
            z = ((1 - h) * moved + h * first).reshape(rows * cols, -1)
            z = settled(sp, z, [first.reshape(rows * cols, -1), moved.reshape(rows * cols, -1)], [h, 1 - h])
        z = looked(name, sp, z, k)
        return decode(sp, z, (rows, cols))

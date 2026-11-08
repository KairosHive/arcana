"""The material's own pixels laid into a picture: no model, nothing invented.

Each mode rebuilds the picture from the texture photograph in its own way: displaced along the
picture's slopes, relit by its relief, its colour or light swapped, its edges traded, combed along its
contours, quilted or matched patch by patch. Ported from latent-lab's composite.py, where the modes
were measured against the photographs they were tuned on.
"""
import math
import pathlib

import numpy as np
from PIL import Image, ImageDraw

LUM = np.array([0.299, 0.587, 0.114], np.float32)


def frame_dims(width, height, long_side, multiple=1):
    """The frame a picture keeps its shape in: its long side at `long_side`, rounded to `multiple`."""
    k = long_side / max(width, height)
    w = max(multiple, int(round(width * k / multiple)) * multiple)
    h = max(multiple, int(round(height * k / multiple)) * multiple)
    return w, h


def open_rgb(path, width=None, height=None):
    """A photograph as RGB. Given a target frame, a JPEG decodes at 1/2, 1/4 or 1/8 scale when that
    still covers it: a 6000px frame read for a 384px render need not be decoded whole."""
    im = Image.open(path)
    if width and height:
        im.draft("RGB", (width, height))
    return im.convert("RGB")


def cover(im, width, height):
    """Crop a picture about its centre to the frame's shape, then resize: it covers the frame."""
    target = width / height
    if im.width / im.height > target:
        keep = max(1, int(round(im.height * target)))
        left = (im.width - keep) // 2
        im = im.crop((left, 0, left + keep, im.height))
    else:
        keep = max(1, int(round(im.width / target)))
        top = (im.height - keep) // 2
        im = im.crop((0, top, im.width, top + keep))
    return im.resize((width, height), Image.LANCZOS)


def fit(path, size):
    """The old square frame, still used by the sheet tools."""
    return cover(open_rgb(path, size, size), size, size)


def arr(im):
    return np.asarray(im, np.float32) / 255.0


def img(a):
    return Image.fromarray((np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8))


def blur(a, sigma):
    """Gaussian blur by FFT on the even extension, so edges reflect instead of wrapping.

    Real-input transforms in single precision, every channel in one call: a slider that blurs on
    each frame cannot afford complex doubles, three times over.
    """
    if sigma <= 0:
        return a.copy()
    h, w = a.shape[:2]
    x = np.asarray(a, np.float32)
    if h * w > 2_000_000 and sigma >= 6:
        # A wide Gaussian carries no fine detail, so on an export-sized frame it is computed on a
        # box-shrunk copy and brought back up: same picture, a fraction of the transform.
        f = int(min(sigma // 3, 16))
        if f >= 2:
            small = _resize(x, max(1, w // f), max(1, h // f), Image.BOX)
            # Bicubic, not bilinear: relief, displace and flow take the GRADIENT of this field, and a
            # bilinear surface has a gradient that jumps at every cell edge, which shows as blocks.
            return _resize(blur(small, sigma / f), w, h, Image.BICUBIC)
    # Mirror-padded by 4 sigma a side, out to a length whose only factors are 2, 3 and 5. An FFT
    # over a prime length (683, from a 1024px frame of a 3:2 picture) runs several times slower;
    # the wrap-around lands in mirrored margin a Gaussian's tail never reaches.
    pad = int(math.ceil(4.0 * sigma)) + 2
    H, W = _fast_len(h + 2 * pad), _fast_len(w + 2 * pad)
    y0, x0 = (H - h) // 2, (W - w) // 2
    widths = [(y0, H - h - y0), (x0, W - w - x0)] + ([(0, 0)] if x.ndim == 3 else [])
    big = np.pad(x, widths, mode="symmetric")
    fy = np.fft.fftfreq(H).astype(np.float32)[:, None]
    fx = np.fft.rfftfreq(W).astype(np.float32)[None, :]
    g = np.exp(-2.0 * (np.pi * sigma) ** 2 * (fx**2 + fy**2)).astype(np.float32)
    if x.ndim == 3:
        g = g[..., None]
    out = np.fft.irfft2(np.fft.rfft2(big, axes=(0, 1)) * g, s=(H, W), axes=(0, 1))
    return out[y0 : y0 + h, x0 : x0 + w].astype(np.float32)


def _fast_len(n):
    """The first length at or above n whose only prime factors are 2, 3 and 5."""
    while True:
        m = n
        for p in (2, 3, 5):
            while m % p == 0:
                m //= p
        if m == 1:
            return n
        n += 1


def _resize(a, width, height, resample):
    if a.ndim == 2:
        return np.asarray(Image.fromarray(a, "F").resize((width, height), resample), np.float32)
    return np.stack([_resize(a[..., c], width, height, resample) for c in range(a.shape[2])], -1)


def bands(content, texture, split, grain=1.0):
    """Content below the split, texture above it."""
    return blur(content, split) + grain * (texture - blur(texture, split))


def sample(tex, X, Y):
    """The texture read at fractional coordinates, bilinear."""
    h, w = tex.shape[:2]
    X = np.clip(X, 0.0, w - 1.001)
    Y = np.clip(Y, 0.0, h - 1.001)
    x0 = np.floor(X).astype(np.int32)
    y0 = np.floor(Y).astype(np.int32)
    fx = (X - x0)[..., None]
    fy = (Y - y0)[..., None]
    return (
        tex[y0, x0] * (1 - fx) * (1 - fy)
        + tex[y0, x0 + 1] * fx * (1 - fy)
        + tex[y0 + 1, x0] * (1 - fx) * fy
        + tex[y0 + 1, x0 + 1] * fx * fy
    )


def displace(content, texture, amp, reach=3.0, shade=0.7):
    """The texture warped by the content's gradients and lit by its light.

    `reach` is the blur before the gradient is taken: small reads pores, large reads features. The
    RMS displacement is normalised to `amp` of the frame, so the knob means the same thing on a
    soft portrait and a hard facade.
    """
    h, w = content.shape[:2]
    span = max(h, w)
    lum = content @ LUM
    struct = blur(lum, reach)
    gy, gx = np.gradient(struct)
    rms = np.sqrt((gx**2 + gy**2).mean()) + 1e-8
    Y, X = np.mgrid[0:h, 0:w].astype(np.float32)
    warped = sample(texture, X + gx / rms * amp * span, Y + gy / rms * amp * span)
    light = blur(lum, 2.0)
    light = light / (light.mean() + 1e-8)
    return warped * (light[..., None] ** shade)


def solve(gx, gy):
    """An image whose gradients are these, by FFT Poisson on the even extension."""
    h, w = gx.shape
    div = (gx - np.roll(gx, 1, axis=1)) + (gy - np.roll(gy, 1, axis=0))
    big = np.block([[div, div[:, ::-1]], [div[::-1, :], div[::-1, ::-1]]])
    H, W = big.shape
    ky = 2.0 * np.cos(2.0 * np.pi * np.arange(H) / H) - 2.0
    kx = 2.0 * np.cos(2.0 * np.pi * np.arange(W) / W) - 2.0
    denom = ky[:, None] + kx[None, :]
    denom[0, 0] = 1.0
    hat = np.fft.fft2(big) / denom
    hat[0, 0] = 0.0
    return np.fft.ifft2(hat).real[:h, :w].astype(np.float32)


def grad(a):
    return np.diff(a, axis=1, append=a[:, -1:]), np.diff(a, axis=0, append=a[-1:, :])


def relief(content, texture, angle, ambient=0.55, reach=2.5):
    """The texture lit by the picture's slopes: a sun the knob walks around the frame.

    The content's luminance is read as height, its gradient as the surface normal, and the texture
    is shaded by how each point faces the light. Nothing of the content's colour survives — only
    its topography, pressed into the material as light.
    """
    lum = blur(content @ LUM, reach)
    gy, gx = np.gradient(lum)
    rms = np.sqrt((gx**2 + gy**2).mean()) + 1e-8
    shade = (np.cos(angle) * gx + np.sin(angle) * gy) / rms
    return texture * np.clip(ambient + 0.45 * shade, 0.0, 1.7)[..., None]


def veins(content, texture, depth, reach=3.0):
    """The texture's own crack graph, carved into the picture.

    Dark lines are read out of the TEXTURE (what sits below its local surround) and drawn across
    the content as engraved shadow — one material's skeleton on another's flesh. `depth` is how
    deep the engraving cuts.
    """
    t = texture @ LUM
    crack = np.clip(blur(t, reach) - t, 0.0, None)
    crack = np.clip(crack / (np.quantile(crack, 0.995) + 1e-8), 0.0, 1.0) ** 1.2
    return content * np.clip(1.0 - depth * crack, 0.0, 1.0)[..., None]


def flow(content, texture, length, reach=3.0, smear=0.35, steps=8):
    """The texture combed along the picture's contours, with its grain kept.

    A tangent field is taken perpendicular to the content's gradients and each pixel WALKS it —
    the field re-read at every step, so the path curves with the form — and reads the texture once
    at where it lands. A single advected sample is a warp, which keeps every grain of the material;
    `smear` mixes in the average over the whole walk, which is the soft streaking of classic line
    integral convolution. 0 is all comb, 1 is all smear; granularity is the dial, not the price.
    `length` is how far the comb pulls, as a fraction of the frame.
    """
    h, w = content.shape[:2]
    lum = blur(content @ LUM, reach)
    gy, gx = np.gradient(lum)
    mag = np.hypot(gx, gy) + 1e-8
    tx, ty = -gy / mag, gx / mag
    Y, X = np.mgrid[0:h, 0:w].astype(np.float32)
    step = length * max(h, w) / max(steps, 1)
    px, py = X.copy(), Y.copy()
    acc = texture.copy()
    for _ in range(steps):
        ix = np.clip(px, 0, w - 1).astype(np.int32)
        iy = np.clip(py, 0, h - 1).astype(np.int32)
        px += tx[iy, ix] * step
        py += ty[iy, ix] * step
        acc += sample(texture, px, py)
    combed = sample(texture, px, py)
    return (1.0 - smear) * combed + smear * (acc / (steps + 1))


def quilt(content, texture, alpha, block=48, candidates=400, seed=0, mapping=False):
    """Texture transfer by quilting (Efros–Freeman): the picture rebuilt from PATCHES of the texture.

    Every block of the output is a verbatim piece of the texture photograph, chosen so its light
    matches the content's light there (`alpha` is how hard that correspondence is pressed against
    seam agreement), and sewn to its neighbours along the minimum-error cut, so the joins hide in
    the material's own grain. Full granularity by construction — every pixel is a real texture
    pixel — and provenance intact. With `mapping`, it returns instead which texture pixel each
    output pixel copies, so `remap` can rebuild the same mosaic from a bigger copy of the texture.
    """
    from numpy.lib.stride_tricks import sliding_window_view

    height, width = content.shape[:2]
    overlap = max(block // 4, 4)
    step = block - overlap
    lum_c = blur(content @ LUM, 1.5)
    lum_t = texture @ LUM
    lum_t = (lum_t - lum_t.mean()) * (lum_c.std() / (lum_t.std() + 1e-8)) + lum_c.mean()
    rng = np.random.default_rng(seed)
    out = np.zeros_like(content)
    out_lum = np.zeros_like(lum_c)
    src_y = np.zeros((height, width), np.int32)
    src_x = np.zeros((height, width), np.int32)

    def seam(err, horizontal):
        # The classic dynamic programme: the cheapest path through the overlap's error surface.
        e = err.T if horizontal else err
        cost = e.copy()
        for r in range(1, len(cost)):
            left = np.roll(cost[r - 1], 1)
            right = np.roll(cost[r - 1], -1)
            left[0] = np.inf
            right[-1] = np.inf
            cost[r] += np.minimum(cost[r - 1], np.minimum(left, right))
        cut = np.zeros(len(cost), np.int32)
        cut[-1] = int(cost[-1].argmin())
        for r in range(len(cost) - 2, -1, -1):
            lo = max(cut[r + 1] - 1, 0)
            cut[r] = lo + int(cost[r][lo : cut[r + 1] + 2].argmin())
        keep = np.arange(e.shape[1])[None, :] >= cut[:, None]  # True where the NEW block wins
        return keep.T if horizontal else keep

    for by in range(0, max(height - overlap, 1), step):
        for bx in range(0, max(width - overlap, 1), step):
            bh, bw = min(block, height - by), min(block, width - bx)
            windows = sliding_window_view(lum_t, (bh, bw))
            ys = rng.integers(0, windows.shape[0], candidates)
            xs = rng.integers(0, windows.shape[1], candidates)
            cand = windows[ys, xs]
            want = ((cand - lum_c[by : by + bh, bx : bx + bw]) ** 2).mean((1, 2))
            join = np.zeros(candidates, np.float32)
            if bx:
                join += ((cand[:, :, :overlap] - out_lum[by : by + bh, bx : bx + overlap]) ** 2).mean((1, 2))
            if by:
                join += ((cand[:, :overlap, :] - out_lum[by : by + overlap, bx : bx + bw]) ** 2).mean((1, 2))
            best = int((alpha * want + (1 - alpha) * join).argmin())
            y0, x0 = int(ys[best]), int(xs[best])
            piece = texture[y0 : y0 + bh, x0 : x0 + bw]
            mask = np.ones((bh, bw), bool)
            if bx:
                err = (lum_t[y0 : y0 + bh, x0 : x0 + overlap] - out_lum[by : by + bh, bx : bx + overlap]) ** 2
                mask[:, :overlap] &= seam(err, horizontal=False)
            if by:
                err = (lum_t[y0 : y0 + overlap, x0 : x0 + bw] - out_lum[by : by + overlap, bx : bx + bw]) ** 2
                mask[:overlap, :] &= seam(err, horizontal=True)
            region = out[by : by + bh, bx : bx + bw]
            region[mask] = piece[mask]
            out_lum[by : by + bh, bx : bx + bw][mask] = lum_t[y0 : y0 + bh, x0 : x0 + bw][mask]
            gy, gx = np.mgrid[y0 : y0 + bh, x0 : x0 + bw]
            src_y[by : by + bh, bx : bx + bw][mask] = gy[mask]
            src_x[by : by + bh, bx : bx + bw][mask] = gx[mask]
    return (src_y, src_x) if mapping else out


def analogy(content, texture, patch=7, melt=1.0, rounds=8, seed=0, mapping=False):
    """Per-pixel texture transfer by PatchMatch — quilting's modern sibling, no blocks, no seams.

    Every output pixel is read from the texture at the place whose NEIGHBOURHOOD of light best
    matches the picture's neighbourhood there, found by the PatchMatch dance: good matches
    propagate to their neighbours, random probes keep the search honest. Solved at half size and
    refined at full, so it costs seconds, not minutes. `patch` is the neighbourhood's side — small
    reads pointillist grain, large reads structure. `melt` smooths the match field afterwards,
    melting the mosaic toward a continuous warp; at 0 every pixel is a verbatim texture pixel.
    """
    r = max(1, int(patch) // 2)
    lc = blur(content @ LUM, 0.8)
    lt = texture @ LUM
    lt = (lt - lt.mean()) * (lc.std() / (lt.std() + 1e-8)) + lc.mean()
    rng = np.random.default_rng(seed)

    def solve(c, t, ny, nx, rounds):
        H, W = c.shape
        Th, Tw = t.shape
        pc = np.pad(c, r, mode="reflect")
        pt = np.pad(t, r, mode="reflect")
        Y, X = np.mgrid[0:H, 0:W]
        offs = [(dy, dx) for dy in range(-r, r + 1) for dx in range(-r, r + 1)]

        def cost(cy, cx):
            got = np.zeros((H, W), np.float32)
            for dy, dx in offs:
                got += (pc[Y + dy + r, X + dx + r] - pt[cy + dy + r, cx + dx + r]) ** 2
            return got

        best = cost(ny, nx)
        for rd in range(rounds):
            s = 1 if rd % 2 == 0 else -1
            wy, wx = max(2, Th >> (rd + 1)), max(2, Tw >> (rd + 1))
            cands = [
                (np.roll(ny, s, 1), np.clip(np.roll(nx, s, 1) + s, 0, Tw - 1)),  # follow a neighbour
                (np.clip(np.roll(ny, s, 0) + s, 0, Th - 1), np.roll(nx, s, 0)),
                (np.clip(ny + rng.integers(-wy, wy + 1, ny.shape), 0, Th - 1),   # look nearby
                 np.clip(nx + rng.integers(-wx, wx + 1, nx.shape), 0, Tw - 1)),
                (rng.integers(0, Th, ny.shape), rng.integers(0, Tw, nx.shape)),  # look anywhere
            ]
            for cy, cx in cands:
                c2 = cost(cy, cx)
                win = c2 < best
                ny, nx = np.where(win, cy, ny), np.where(win, cx, nx)
                best = np.where(win, c2, best)
        return ny, nx

    small_c, small_t = lc[::2, ::2], lt[::2, ::2]
    ny = rng.integers(0, small_t.shape[0], small_c.shape)
    nx = rng.integers(0, small_t.shape[1], small_c.shape)
    ny, nx = solve(small_c, small_t, ny, nx, rounds)
    Th, Tw = lt.shape
    ny = np.clip(np.repeat(np.repeat(ny, 2, 0), 2, 1)[: lc.shape[0], : lc.shape[1]] * 2, 0, Th - 1)
    nx = np.clip(np.repeat(np.repeat(nx, 2, 0), 2, 1)[: lc.shape[0], : lc.shape[1]] * 2, 0, Tw - 1)
    ny, nx = solve(lc, lt, ny, nx, 2)
    if mapping:
        return ny, nx
    if melt > 0:
        return sample(texture, blur(nx.astype(np.float32), melt), blur(ny.astype(np.float32), melt))
    return texture[ny, nx]


def remap(map_y, map_x, texture, out_h, out_w, melt=0.0):
    """Carry a correspondence found at a working size up to any size, reading the FULL texture.

    The map says which texture pixel each working pixel copies. What varies slowly is its OFFSET
    (where it reads, minus where it sits), so the offsets are scaled up and added to full-size
    positions. Every exported pixel is then a real pixel of the full-size texture — with grain the
    working size never saw — while the arrangement is the one solved small. `texture` must cover
    the same frame as the working texture did, at the output size.
    """
    h, w = map_y.shape
    sy, sx = out_h / h, out_w / w
    oy = (map_y.astype(np.float32) - np.arange(h, dtype=np.float32)[:, None]) * sy
    ox = (map_x.astype(np.float32) - np.arange(w, dtype=np.float32)[None, :]) * sx
    Y, X = np.mgrid[0:out_h, 0:out_w].astype(np.float32)
    if melt > 0:
        iy = np.minimum((np.arange(out_h) / sy).astype(np.int32), h - 1)
        ix = np.minimum((np.arange(out_w) / sx).astype(np.int32), w - 1)
        return sample(texture, X + blur(ox[iy][:, ix], melt * sx), Y + blur(oy[iy][:, ix], melt * sy))
    # Each output pixel reads the texture once through each of its four nearest working pixels'
    # offsets and blends the readings bilinearly. Where the four agree — inside every patch — all
    # four reads are the same texture pixel, verbatim. Only where patches meet do the readings
    # differ, and the seam becomes a crossfade one working pixel wide instead of a staircase.
    th, tw = texture.shape[:2]
    fy = np.clip((np.arange(out_h, dtype=np.float32) + 0.5) / sy - 0.5, 0, h - 1)
    fx = np.clip((np.arange(out_w, dtype=np.float32) + 0.5) / sx - 0.5, 0, w - 1)
    y0, x0 = fy.astype(np.int32), fx.astype(np.int32)
    y1, x1 = np.minimum(y0 + 1, h - 1), np.minimum(x0 + 1, w - 1)
    wy, wx = (fy - y0)[:, None, None], (fx - x0)[None, :, None]
    out = np.zeros((out_h, out_w, texture.shape[2]), np.float32)
    for ry, rx, weight in ((y0, x0, (1 - wy) * (1 - wx)), (y0, x1, (1 - wy) * wx),
                           (y1, x0, wy * (1 - wx)), (y1, x1, wy * wx)):
        ty = np.clip(np.rint(Y + oy[ry][:, rx]), 0, th - 1).astype(np.int32)
        tx = np.clip(np.rint(X + ox[ry][:, rx]), 0, tw - 1).astype(np.int32)
        out += texture[ty, tx] * weight
    return out


def guided(content, texture, radius, eps=0.015, detail=0.35):
    """The texture's colour poured into the picture's structure: a guided filter, not a blur.

    Each texture channel is re-expressed as a local linear function of the content's luminance, so
    its tones redistribute along the picture's own edges — crisp where the guide is crisp, soft
    where it is flat. `radius` is the neighbourhood the fit sees; `detail` returns the texture's
    own grain on top, so structure comes from one photograph and surface from the other.
    """
    guide = content @ LUM
    s = max(float(radius), 1.0) / 2.0
    m_i = blur(guide, s)
    v_i = blur(guide * guide, s) - m_i * m_i
    out = np.empty_like(texture)
    for ch in range(3):
        p = texture[..., ch]
        m_p = blur(p, s)
        a = (blur(guide * p, s) - m_i * m_p) / (v_i + eps)
        b = m_p - a * m_i
        out[..., ch] = blur(a, s) * guide + blur(b, s)
    return out + detail * (texture - blur(texture, s))


def chroma(content, texture, share):
    """The content's light wearing the texture's colour: Y from one, CbCr from the other.

    `share` lets some of the texture's own luminance back in, which returns its grain without
    returning its shapes.
    """
    yc = content @ LUM
    yt = texture @ LUM
    y = (1 - share) * yc + share * yt
    cb = texture @ np.array([-0.168736, -0.331264, 0.5], np.float32)
    cr = texture @ np.array([0.5, -0.418688, -0.081312], np.float32)
    return np.stack([y + 1.402 * cr, y - 0.344136 * cb - 0.714136 * cr, y + 1.772 * cb], -1)


def mask(content, texture, level, feather=0.08):
    """Luminance-keyed compositing: the texture floods the content from its shadows upward.

    `level` is the waterline — at 0.3 only the darkest pools have turned to texture, at 0.9 only
    the highlights still hold. The edge is feathered in luminance, not in space, so it follows the
    picture's own forms.
    """
    lum = blur(content @ LUM, 1.5)
    m = 1.0 / (1.0 + np.exp((lum - level) / max(feather, 1e-3)))
    return content * (1 - m[..., None]) + texture * m[..., None]


def blend(content, texture, wmix):
    """Mixed-gradient Poisson: the stronger gradient wins, judged on luminance, kept per channel."""
    gcx, gcy = grad(content @ LUM)
    gtx, gty = grad(texture @ LUM)
    pick = (wmix * np.hypot(gcx, gcy)) >= ((1.0 - wmix) * np.hypot(gtx, gty))
    out = []
    for c in range(3):
        cx, cy = grad(content[..., c])
        tx, ty = grad(texture[..., c])
        u = solve(np.where(pick, cx, tx), np.where(pick, cy, ty))
        base = wmix * content[..., c].mean() + (1 - wmix) * texture[..., c].mean()
        out.append(u + base)
    return np.stack(out, -1)



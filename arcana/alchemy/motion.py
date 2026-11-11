"""A moving-camera exposure (intentional camera movement) made from a still picture.

A real exposure with the camera moving is the light of the scene summed over every place the camera
passed. This sums copies of the picture along a camera path in linear light, so a flame or a glint
draws a bright streak while the shadows around it blur into each other, as on film.

The path, in one exposure:

    sweep       how far the camera travels, as a share of the frame's long side, toward `direction`
                (0 rad runs right, pi/2 runs down, so a vertical sweep is pi/2)
    rotation    degrees the camera turns about the frame's centre
    zoom        how much the frame grows (above 0) or shrinks (below 0) by the end, about the centre
    wobble      a sideways oscillation across the sweep, as a share of the frame, `cycles` times

and how the time of the exposure is spent along it:

    hold start  the share of the exposure spent still before moving: a sharp impression of the picture
    hold end    the share spent still at the end of the path: a second, displaced impression
    centred     the path runs through the picture's own place (else it starts there)

`trails` moves the sum from the average of the copies (a single exposure) toward the brightest copy at
each place (several exposures laid with Lighten), which keeps streaks of light at full strength.
`focus pull` blurs the copies more the further along the path they are, as when the focus ring turns
during the exposure.

Three ways of layering, from the practice of combining ICM with multiple exposures:

    echoes      a few separate exposures stepped along the path, each fainter, in place of one streak
    strokes     several movements fanned across a spread of directions, as crossing brushstrokes
    glow        a soft copy screened over the result (the Orton effect)
"""

from __future__ import annotations

import math

import numpy as np

GAMMA = 2.2


def _device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def path(steps: int, hold_start: float, hold_end: float):
    """(u, weight) for each copy: u in 0..1 along the path, weights summing to 1. The holds put their
    share of the exposure on the path's ends."""
    steps = max(2, int(steps))
    hs, he = max(float(hold_start), 0.0), max(float(hold_end), 0.0)
    if hs + he > 0.98:
        k = 0.98 / (hs + he)
        hs, he = hs * k, he * k
    u = np.linspace(0.0, 1.0, steps)
    w = np.full(steps, (1.0 - hs - he) / steps)
    w[0] += hs
    w[-1] += he
    return u, w


def _gauss(t, sigma):
    """A [1, C, H, W] tensor blurred by sigma pixels, separably, with reflected edges."""
    import torch
    import torch.nn.functional as F

    r = max(1, int(math.ceil(sigma * 3)))
    x = torch.arange(-r, r + 1, device=t.device, dtype=torch.float32)
    g = torch.exp(-(x * x) / (2 * sigma * sigma))
    n = t.shape[1]
    g = (g / g.sum()).view(1, 1, 1, -1).repeat(n, 1, 1, 1)
    out = F.conv2d(F.pad(t, (r, r, 0, 0), mode="reflect"), g, groups=n)
    return F.conv2d(F.pad(out, (0, 0, r, r), mode="reflect"), g.transpose(2, 3), groups=n)


def _disc(t, radius):
    """A [1, C, H, W] tensor through a round lens blur of `radius` pixels (a soft-edged disc)."""
    import torch
    import torch.nn.functional as F

    r = int(math.ceil(radius)) + 1
    y, x = torch.meshgrid(torch.arange(-r, r + 1, device=t.device, dtype=torch.float32),
                          torch.arange(-r, r + 1, device=t.device, dtype=torch.float32), indexing="ij")
    k = (radius + 0.5 - torch.hypot(x, y)).clamp(0, 1)
    k = (k / k.sum())[None, None].repeat(t.shape[1], 1, 1, 1)
    return F.conv2d(F.pad(t, (r, r, r, r), mode="reflect"), k, groups=t.shape[1])


def _grads(a):
    """Central differences of a [H, W] tensor, edges repeated."""
    import torch.nn.functional as F

    p = F.pad(a[None, None], (1, 1, 1, 1), mode="replicate")[0, 0]
    return (p[1:-1, 2:] - p[1:-1, :-2]) / 2, (p[2:, 1:-1] - p[:-2, 1:-1]) / 2


def expose(c, sweep=0.15, direction=math.pi / 2, rotation=0.0, zoom=0.0, wobble=0.0, cycles=3.0,
           hold_start=0.0, hold_end=0.0, centred=True, trails=0.0, focus_pull=0.0, steps=48,
           echoes=0, fade=0.0, strokes=1, spread=0.0, glow=0.0, echo_step=0.06, echo_angle=math.pi / 2,
           irregular=0.0, drift=0.0, seed=0, detail=0.0, depth=None, parallax=0.0, follow=0.0, anchor=0.0,
           light=0.0, defocus=0.0, pivot_x=0.5, pivot_y=0.5, hold=None):
    """The picture c ([H, W, 3] floats in 0..1) exposed along the path. `focus_pull` is in pixels.

    `echoes` (2 or more) repeats the whole movement that many times, each repeat stepped `echo_step` (a
    share of the frame) further along `echo_angle` and `fade` weaker than the one before: several
    exposures of one gesture, as ridges laid one behind another. Exact repeats at equal steps read as
    mechanical, so `irregular` spaces them unevenly and `drift` moves each a little across its step and
    in scale; `seed` picks how. `strokes` splits the exposure into that many movements fanned across
    `spread` radians about `direction`, as gestures layered in one frame. `glow` screens a soft copy of
    the result over it, the dreamy halo of a sharp and a blurred exposure laid together.

    Two keep the picture's structure through the movement. `detail` lays the picture's own fine detail
    (what a few pixels' blur removes) back over the exposure, so its edges stay while its light and
    colour flow. `parallax`, given a `depth` map ([H, W], near is 1), moves near places further than
    far ones, as a moving camera does: at 1 the farthest places hold still.

    Three bend the movement place by place, so the streaks belong to the picture. `anchor` holds the
    strong edges still and lets flat places flow, in proportion to the edge. `light` lets bright places
    streak and dark ones hold, as lights dominate a real long exposure. `follow` bends the sweep along
    the picture's own contours (its structure tensor), so the strokes run with its forms. With either
    `anchor` or `follow` each copy is marched step by step along that field, rather than shifted whole,
    so neighbouring places stay together and nothing tears. `light` splits the picture in two instead:
    its light (each place weighted by its brightness) moves the whole way and streaks across whatever it
    passes, while the rest holds, as a flash at the end of a long exposure freezes the dark subject
    inside the trails.

    `defocus` softens the whole picture through a round lens blur of that radius in pixels, in linear
    light, so lights bloom into discs, before the camera moves. `pivot_x` and `pivot_y` (0..1 of the
    frame) place the centre of a turn or a zoom, so a swirl can turn about a figure. A negative `trails`
    keeps the darkest copy at each place instead of the brightest, so dark branches streak across a pale
    sky.

    `hold` ([H, W] in 0..1, a feathered mask) names places that stay still: the movement slows to
    nothing inside them and flows on around them, the defocus spares them, and they are laid back sharp
    at the end, so a silhouette stands clear in a moving, softened world."""
    import torch
    import torch.nn.functional as F

    H, W = c.shape[:2]
    dev = _device()
    lin = torch.as_tensor(np.ascontiguousarray(np.clip(c, 0, 1)), dtype=torch.float32, device=dev)
    lin = lin.pow(GAMMA).permute(2, 0, 1)[None]                  # [1, 3, H, W], linear light
    long = max(H, W)
    sharp = lin
    held = None
    if hold is not None:
        held = torch.as_tensor(np.asarray(hold, np.float32), device=dev)
        if held.shape != (H, W):
            held = F.interpolate(held[None, None], size=(H, W), mode="bilinear", align_corners=False)[0, 0]
        held = held.clamp(0, 1)[None, None]
    if defocus >= 0.5:
        lin = _disc(lin, float(defocus))
    echoes, strokes = int(echoes or 0), max(1, int(strokes or 1))
    # The farthest corner from the turn's centre moves most in a turn or a zoom.
    reach = max(math.hypot(cx * W, cy * H) for cx in (pivot_x, 1 - pivot_x) for cy in (pivot_y, 1 - pivot_y))
    # Copies close enough that no streak breaks into dashes: about a pixel apart wherever the frame moves
    # most (its corners, for a turn or a zoom). `steps` is the least.
    travel = (abs(sweep) * long + abs(math.radians(rotation)) * reach + abs(zoom) * reach
              + abs(wobble) * long * 2 * math.pi * max(cycles, 0.0))
    reps = echoes if echoes >= 2 else 1
    n = int(min(max(int(steps), math.ceil(travel / 0.9) + 1), max(720 // (strokes * reps), 2)))
    u, w = path(n, hold_start, hold_end)
    # Each repeat of the movement: its offset (x, y in pixels from its step and drift), its scale, its weight.
    repeats = [(0.0, 0.0, 1.0, 1.0)]
    if reps > 1:
        rng = np.random.default_rng(int(seed))
        gap = 1.0 / (reps - 1)
        at = np.linspace(-0.5, 0.5, reps) + float(irregular) * gap * (rng.random(reps) - 0.5)
        across = float(drift) * (rng.random(reps) - 0.5) * 2
        kicks = 1.0 + float(drift) * (rng.random(reps) - 0.5) * 2
        ew = (1.0 - min(max(float(fade), 0.0), 0.95)) ** np.arange(reps)
        ew = ew / ew.sum()
        ex, ey = math.cos(echo_angle), math.sin(echo_angle)
        repeats = [(float(ex * at[i] * echo_step * (reps - 1) - ey * across[i]) * long,
                    float(ey * at[i] * echo_step * (reps - 1) + ex * across[i]) * long,
                    float(kicks[i]), float(ew[i])) for i in range(reps)]
    # Normalised grid coordinates (-1..1 on each axis); a pixel shift of d is 2d/W across, 2d/H down.
    ys, xs = torch.meshgrid(torch.linspace(-1, 1, H, device=dev), torch.linspace(-1, 1, W, device=dev),
                            indexing="ij")
    px, py = xs * (W / 2), ys * (H / 2)                          # pixel offsets from the centre
    cx0, cy0 = (float(pivot_x) - 0.5) * W, (float(pivot_y) - 0.5) * H   # where a turn or zoom centres
    reach_of = 1.0                                               # how far each place moves with the camera
    if depth is not None and parallax > 0:
        d = torch.as_tensor(np.asarray(depth, np.float32), device=dev)
        if d.shape != (H, W):
            d = F.interpolate(d[None, None], size=(H, W), mode="bilinear", align_corners=False)[0, 0]
        d = (d - d.min()) / (d.max() - d.min() + 1e-6)
        reach_of = (1.0 - float(parallax)) + float(parallax) * d
    flowing = follow > 0 or anchor > 0 or held is not None
    lightw = None
    if light > 0:
        still0 = lin.pow(1 / GAMMA)
        lum0 = (still0[0] * torch.tensor([0.299, 0.587, 0.114], device=dev)[:, None, None]).sum(0)
        lv = _gauss(lum0[None, None], max(1.0, 0.002 * long))[0, 0]
        flat = lv.flatten()
        top_light = torch.quantile(flat[:: max(1, flat.numel() // 2_000_000)], 0.999)   # the lights, not the sky
        lightw = ((lv / (top_light + 1e-6)).clamp(0, 1) ** 2)[None, None]
        lit_src = lin * lightw
    if flowing:
        still = lin.pow(1 / GAMMA)
        lum = (still[0] * torch.tensor([0.299, 0.587, 0.114], device=dev)[:, None, None]).sum(0)
        gx, gy = _grads(_gauss(lum[None, None], 1.0)[0, 0])
        reach = torch.ones_like(lum)
        if anchor > 0:
            e = _gauss(torch.hypot(gx, gy)[None, None], max(1.0, 0.004 * long))[0, 0]
            e = (e / (torch.quantile(e.flatten()[::7], 0.9) + 1e-6)).clamp(0, 1)
            reach = reach * (1.0 - float(anchor) * e)
        if not isinstance(reach_of, float):
            reach = reach * reach_of
        if held is not None:
            reach = reach * (1.0 - held[0, 0])
        if follow > 0:
            sig = max(2.0, 0.012 * long)
            jxx = _gauss((gx * gx)[None, None], sig)[0, 0]
            jxy = _gauss((gx * gy)[None, None], sig)[0, 0]
            jyy = _gauss((gy * gy)[None, None], sig)[0, 0]
            theta = 0.5 * torch.atan2(2 * jxy, jxx - jyy) + math.pi / 2     # along the contours
            tx, ty = torch.cos(theta), torch.sin(theta)
            coherence = (torch.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / (jxx + jyy + 1e-8)).clamp(0, 1)

    def field(dx, dy):
        """[1, 2, H, W]: how far and which way each place moves per pixel of sweep."""
        fx, fy = torch.full_like(px, dx), torch.full_like(px, dy)
        if follow > 0:
            sign = torch.sign(tx * dx + ty * dy)
            sign = torch.where(sign == 0, torch.ones_like(sign), sign)
            f = float(follow) * coherence
            fx, fy = (1 - f) * dx + f * tx * sign, (1 - f) * dy + f * ty * sign
            n = torch.hypot(fx, fy).clamp_min(1e-6)
            fx, fy = fx / n, fy / n
        return torch.stack([fx * reach, fy * reach])[None]

    def at(v, x, y):
        """The field v read at pixel offsets (x, y) from the centre."""
        g = torch.stack([x / (W / 2), y / (H / 2)], -1)[None]
        got = F.grid_sample(v, g, mode="bilinear", padding_mode="border", align_corners=True)[0]
        return got[0], got[1]

    def marched(v, offsets):
        """Each place carried along v by each signed distance in pixels: {distance: (x, y)}."""
        out = {o: (px, py) for o in offsets if o == 0}
        for sign in (1, -1):
            x, y, done = px, py, 0.0
            for d in sorted((o for o in offsets if (o > 0 if sign > 0 else o < 0)), key=abs):
                todo = abs(d) - done
                m = max(1, int(math.ceil(todo / 1.0)))
                h = todo / m
                for _ in range(m):                                   # midpoint steps, about a pixel each
                    vx, vy = at(v, x, y)
                    mx, my = x - sign * 0.5 * h * vx, y - sign * 0.5 * h * vy
                    vx, vy = at(v, mx, my)
                    x, y = x - sign * h * vx, y - sign * h * vy
                done = abs(d)
                out[d] = (x, y)
        return out

    blurred = {}

    def soft(sigma):
        """The picture blurred by sigma pixels, kept by half-pixel steps."""
        key = round(sigma * 2) / 2
        if key <= 0.25:
            return lin
        if key not in blurred:
            blurred[key] = _gauss(lin, key)
        return blurred[key]

    acc = torch.zeros_like(lin)
    top = torch.zeros_like(lin)
    bottom = None
    acc_lit = torch.zeros_like(lin) if lightw is not None else None
    for j, (rx, ry, kick, rw) in ((j, r) for j in range(strokes) for r in repeats):
        angle = direction + (spread * ((j + 0.5) / strokes - 0.5) if strokes > 1 else 0.0)
        dx, dy = math.cos(angle), math.sin(angle)
        if flowing:
            dists = [round(sweep * long * (float(ui) - (0.5 if centred else 0.0)), 4) for ui in u]
            paths = marched(field(dx, dy), set(dists))
        for i, (ui, wi) in enumerate(zip(u, w)):
            s = float(ui) - (0.5 if centred else 0.0)            # where along the sweep this copy sits
            shift = sweep * long * s
            side = wobble * long * math.sin(2 * math.pi * cycles * float(ui))
            a = math.radians(rotation * s)
            k = max((1.0 + zoom * s) * kick, 0.05)
            if flowing:
                bx, by = paths[dists[i]]
                ox, oy = -dy * side + rx, dx * side + ry
                qx, qy = (bx - cx0 - ox) / k, (by - cy0 - oy) / k
            else:
                ox = dx * shift - dy * side + rx
                oy = dy * shift + dx * side + ry
                # The camera moved by (ox, oy), turned by a and zoomed by k: a scene point lands where the
                # inverse of that motion takes it, so each output pixel reads the picture there.
                qx, qy = (px - cx0 - ox * reach_of) / k, (py - cy0 - oy * reach_of) / k
            sx = math.cos(a) * qx + math.sin(a) * qy + cx0
            sy = -math.sin(a) * qx + math.cos(a) * qy + cy0
            grid = torch.stack([sx / (W / 2), sy / (H / 2)], -1)[None]
            src = soft(focus_pull * min(max(float(ui), 0.0), 1.0)) if focus_pull > 0 else lin
            copy = F.grid_sample(src, grid, mode="bilinear", padding_mode="reflection", align_corners=True)
            acc += (float(wi) * rw / strokes) * copy
            if acc_lit is not None:
                acc_lit += (float(wi) * rw / strokes) * F.grid_sample(lit_src, grid, mode="bilinear",
                                                                    padding_mode="reflection", align_corners=True)
            if trails > 0:
                top = torch.maximum(top, copy)
            elif trails < 0:
                bottom = copy if bottom is None else torch.minimum(bottom, copy)
    if trails > 0:
        out = (1 - trails) * acc + trails * top
    elif trails < 0 and bottom is not None:
        out = (1 + trails) * acc - trails * bottom
    else:
        out = acc
    if acc_lit is not None:
        # The light moves the whole way; the rest holds, or moves only as the edges and forms let it.
        base = acc if flowing else lin
        out = (1 - float(light)) * out + float(light) * (base * (1 - lightw) + acc_lit)
    if held is not None:
        out = held * sharp + (1 - held) * out                    # the held places laid back sharp
    out = out.clamp(0, 1).pow(1 / GAMMA)
    if detail > 0:
        still = lin.pow(1 / GAMMA)
        out = out + float(detail) * (still - _gauss(still, max(1.5, 0.003 * long)))
    if glow > 0:
        halo = _gauss(out, 0.02 * long)
        out = 1 - (1 - out) * (1 - min(float(glow), 1.0) * halo)
    return out.clamp(0, 1)[0].permute(1, 2, 0).float().cpu().numpy()

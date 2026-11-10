# motion.py -- how a piece of video moves, as numbers and as a line of words
#
# An embedding of sampled frames says what a section shows and next to nothing
# about how it moves (VIDEO_ROADMAP.md, 2.1). This module measures the movement
# directly, from optical flow between consecutive small frames:
#
#   1. Flow between each pair of frames (Farneback, on 96 px grey frames).
#   2. A similarity transform fitted to the flow: where the whole frame goes
#      is the camera (pan, tilt, zoom, roll). What is left over is the subject.
#   3. Per section, the mean and spread of those, plus shake (how much the
#      camera's path jitters), flicker (brightness changing between frames)
#      and how regular the movement is.
#
# The numbers are deterministic, cheap and readable, which is what lets them
# be facets and sliders, a second way to arrange the map, and the cue for a
# match cut. `describe()` turns them into the line shown under a result.
#
# Units. Flow is in fractions of the frame's width per second, so a value means
# the same at any resolution and frame rate: a pan of 0.25 crosses the frame in
# four seconds.

from __future__ import annotations

import numpy as np

SIDE = 96

# One row per pair of frames.
PAIR = ("pan_x", "pan_y", "zoom", "roll", "subject", "energy", "flicker")

# One row per section, in this order: the vector that is stored.
NAMES = ("pan_x", "pan_y", "pan", "zoom", "roll", "subject", "energy", "shake", "flicker",
         "pan_spread", "zoom_spread", "subject_spread", "energy_spread", "regular")

# Where "still" ends, in the units above. Read off the person's own footage with
# evals/video_sections.py: a locked-off shot of a quiet scene sits under these,
# and a deliberate move sits well over them.
STILL = 0.012          # camera travel, widths per second
ZOOMING = 0.02         # scale change per second
MOVING = 0.02          # subject movement, widths per second
FAST = 0.25
SHAKY = 0.06
FLICKERING = 0.04


def grey(frame: np.ndarray, side: int = SIDE) -> np.ndarray:
    """An RGB frame as a small grey one, for the flow."""
    import cv2
    g = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if frame.ndim == 3 else frame
    h, w = g.shape[:2]
    if max(h, w) > side:
        s = side / max(h, w)
        g = cv2.resize(g, (max(8, round(w * s)), max(8, round(h * s))), interpolation=cv2.INTER_AREA)
    return g


def pair(a: np.ndarray, b: np.ndarray, dt: float) -> np.ndarray:
    """One PAIR row: what happened between two grey frames `dt` seconds apart."""
    import cv2
    h, w = a.shape[:2]
    flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    xs -= (w - 1) / 2
    ys -= (h - 1) / 2
    fx, fy = flow[..., 0], flow[..., 1]
    # Only pixels with something to see: flow over a flat sky is noise.
    gx = cv2.Sobel(a, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(a, cv2.CV_32F, 0, 1)
    tex = np.hypot(gx, gy)
    keep = tex > max(float(np.percentile(tex, 60)), 4.0)
    n = int(keep.sum())
    flick = abs(float(b.mean()) - float(a.mean())) / 255.0 / dt
    if n < 30:
        return np.array([0, 0, 0, 0, 0, 0, flick], np.float32)
    x, y, u, v = xs[keep], ys[keep], fx[keep], fy[keep]
    # Similarity fit: u = tx + s*x - r*y, v = ty + r*x + s*y. Its least-squares
    # solution in closed form, since x and y are centred on the frame.
    #
    # Fitted twice. A subject crossing the frame pulls the first fit towards
    # itself and reads as a zoom, so the second fit uses only the two thirds of
    # the pixels the first one explained best: the background, where there is one.
    use = np.ones(len(x), dtype=bool)
    for _ in range(2):
        xu, yu = x[use], y[use]
        one, zero = np.ones_like(xu), np.zeros_like(xu)
        A = np.concatenate([np.stack([one, zero, xu, -yu], 1), np.stack([zero, one, yu, xu], 1)])
        tx, ty, s, r = (float(c) for c in np.linalg.lstsq(
            A, np.concatenate([u[use], v[use]]), rcond=None)[0])
        ru = u - tx - (s * x - r * y)
        rv = v - ty - (r * x + s * y)
        err = np.hypot(ru, rv)
        use = err <= np.percentile(err, 67)
    scale = 1.0 / (w * dt)
    return np.array([
        -tx * scale,                    # the picture moving left is the camera panning right
        ty * scale,                     # the picture moving down is the camera tilting up
        s / dt,                         # growth per second: positive is zooming in
        r / dt,
        float(np.median(np.hypot(ru, rv))) * scale,
        float(np.mean(np.hypot(u, v))) * scale,
        flick,
    ], np.float32)


def pairs(frames, fps: float) -> np.ndarray:
    """PAIR rows for consecutive frames (RGB or grey), one fewer than the frames."""
    gs = [grey(f) for f in frames]
    if len(gs) < 2:
        return np.zeros((0, len(PAIR)), np.float32)
    dt = 1.0 / fps
    return np.stack([pair(gs[i], gs[i + 1], dt) for i in range(len(gs) - 1)])


def summarise(rows: np.ndarray) -> np.ndarray:
    """The NAMES vector for a section, from its PAIR rows."""
    out = np.zeros(len(NAMES), np.float32)
    if len(rows) == 0:
        return out
    R = np.asarray(rows, np.float32)
    c = {k: R[:, i] for i, k in enumerate(PAIR)}
    travel = np.hypot(c["pan_x"], c["pan_y"])
    # Shake is the camera changing its mind: how far the travel swings from
    # one pair to the next, which a steady pan does not do and a hand does.
    steps = np.hypot(np.diff(c["pan_x"]), np.diff(c["pan_y"])) if len(R) > 1 else np.zeros(1)
    e = c["energy"] - c["energy"].mean()
    regular = 0.0
    if len(e) >= 8 and float(e.std()) > 1e-6:
        ac = np.correlate(e, e, "full")[len(e):] / float((e * e).sum())
        regular = float(max(0.0, ac[1:len(e) // 2].max())) if len(ac) > 2 else 0.0
    vals = {
        "pan_x": c["pan_x"].mean(), "pan_y": c["pan_y"].mean(),
        "pan": float(np.hypot(c["pan_x"].mean(), c["pan_y"].mean())),
        "zoom": c["zoom"].mean(), "roll": c["roll"].mean(),
        "subject": float(np.median(c["subject"])), "energy": c["energy"].mean(),
        "shake": float(np.median(steps)), "flicker": float(np.median(c["flicker"])),
        "pan_spread": travel.std(), "zoom_spread": c["zoom"].std(),
        "subject_spread": c["subject"].std(), "energy_spread": c["energy"].std(),
        "regular": regular,
    }
    return np.array([vals[k] for k in NAMES], np.float32)


def as_dict(vec) -> dict:
    return {k: float(v) for k, v in zip(NAMES, vec)}


def describe(vec) -> str:
    """
    A section's movement in a few words: "slow pan right, subject moving".

    Camera first, then the subject, then what the picture itself does. Each
    part is only said when its number clears the line above, so a quiet shot
    reads "still camera" and nothing more.
    """
    m = as_dict(vec)
    parts = []
    cam = []
    steady_travel = m["pan"] > STILL and m["pan"] > 0.6 * m["shake"]
    if steady_travel:
        ax, ay = abs(m["pan_x"]), abs(m["pan_y"])
        speed = "fast " if m["pan"] > FAST else "slow " if m["pan"] < 0.06 else ""
        if ax >= ay:
            cam.append(f"{speed}pan {'right' if m['pan_x'] > 0 else 'left'}")
        else:
            cam.append(f"{speed}tilt {'up' if m['pan_y'] > 0 else 'down'}")
    if abs(m["zoom"]) > ZOOMING:
        cam.append("zoom in" if m["zoom"] > 0 else "zoom out")
    if m["shake"] > SHAKY:
        cam.append("handheld")
    parts.append(", ".join(cam) if cam else "still camera")
    if m["subject"] > MOVING:
        parts.append("subject moving fast" if m["subject"] > 0.12 else "subject moving")
    elif not cam and m["energy"] < STILL:
        parts.append("nothing moves")
    if m["flicker"] > FLICKERING:
        parts.append("flickering")
    # Said sparingly: on the person's footage a looser line called tripod shots of
    # water and trees rhythmic, where the flow of a still scene repeats by chance.
    if m["regular"] > 0.75 and m["energy"] > 4 * STILL:
        parts.append("rhythmic")
    return ", ".join(parts)


def tags(vec) -> list[str]:
    """The same reading as facet values: still, pan, tilt, zoom, handheld, subject, flicker, rhythmic."""
    line = describe(vec)
    out = []
    for word, tag in (("still camera", "still"), ("pan", "pan"), ("tilt", "tilt"), ("zoom", "zoom"),
                      ("handheld", "handheld"), ("subject", "subject"), ("flicker", "flicker"),
                      ("rhythmic", "rhythmic")):
        if word in line:
            out.append(tag)
    return out


# ─────────────────────────── words for movement ───────────────────────────
# Layer 2 of the dynamics channel: a query about movement, scored against the
# numbers. Each concept is a few words a person would type and a function from
# a section's motion to a value between 0 and 1. A query is read for the
# concepts it names, and a section scores the mean of those. Nothing is
# embedded, so every result can say which words it was matched on.

def _sat(x, half):
    """0 at nothing, 0.5 at `half`, towards 1 beyond it."""
    x = np.maximum(np.asarray(x, np.float32), 0.0)
    return x / (x + half)


def _col(V, name):
    return V[:, NAMES.index(name)]


CONCEPTS: tuple = (
    ("pan left", ("pan left", "panning left", "pans left", "moving left", "leftward"),
     lambda V: _sat(-_col(V, "pan_x"), 0.05)),
    ("pan right", ("pan right", "panning right", "pans right", "moving right", "rightward"),
     lambda V: _sat(_col(V, "pan_x"), 0.05)),
    ("tilt up", ("tilt up", "tilting up", "tilts up", "rising", "upward"),
     lambda V: _sat(_col(V, "pan_y"), 0.05)),
    ("tilt down", ("tilt down", "tilting down", "tilts down", "descending", "downward"),
     lambda V: _sat(-_col(V, "pan_y"), 0.05)),
    ("zoom in", ("zoom in", "zooming in", "zooms in", "push in", "dolly in", "approaching"),
     lambda V: _sat(_col(V, "zoom"), 0.04)),
    ("zoom out", ("zoom out", "zooming out", "zooms out", "pull out", "pull back", "dolly out", "receding"),
     lambda V: _sat(-_col(V, "zoom"), 0.04)),
    ("pan", ("pan", "panning", "pans", "travelling", "traveling", "tracking", "sweep", "sweeping"),
     lambda V: _sat(np.abs(_col(V, "pan_x")), 0.05)),
    ("tilt", ("tilt", "tilting", "tilts"),
     lambda V: _sat(np.abs(_col(V, "pan_y")), 0.05)),
    ("zoom", ("zoom", "zooming", "zooms"),
     lambda V: _sat(np.abs(_col(V, "zoom")), 0.04)),
    ("handheld", ("handheld", "hand-held", "hand held", "shaky", "shaking", "shake", "jittery", "unsteady"),
     lambda V: _sat(_col(V, "shake"), SHAKY)),
    ("still", ("still", "static", "locked off", "locked-off", "tripod", "fixed", "steady", "motionless"),
     lambda V: np.exp(-(_col(V, "pan") + _col(V, "shake") + 2 * np.abs(_col(V, "zoom"))) / 0.03)),
    ("fast", ("fast", "quick", "rapid", "frantic", "energetic", "dynamic"),
     lambda V: _sat(_col(V, "energy"), 0.15)),
    ("slow", ("slow", "slowly", "gentle", "gently", "drifting", "drift"),
     lambda V: _sat(_col(V, "energy"), 0.01) * np.exp(-_col(V, "energy") / 0.08)),
    ("calm", ("calm", "quiet", "nothing moves", "tranquil", "serene"),
     lambda V: np.exp(-_col(V, "energy") / 0.02)),
    ("subject moving", ("subject moving", "moving subject", "action", "movement", "moving", "walking",
                        "running", "dancing", "gesture"),
     lambda V: _sat(_col(V, "subject"), 0.05)),
    ("flicker", ("flicker", "flickering", "strobe", "strobing", "flashing", "flash", "pulsing light"),
     lambda V: _sat(_col(V, "flicker"), FLICKERING)),
    ("rhythmic", ("rhythmic", "rhythm", "pulsing", "periodic", "repeating", "looping"),
     lambda V: np.clip(_col(V, "regular"), 0.0, 1.0) * _sat(_col(V, "energy"), STILL)),
)


def vocabulary() -> list[str]:
    """The movements a query can name, for the hint under the box."""
    return [name for name, _, _ in CONCEPTS]


def read_query(text: str) -> list[str]:
    """The concepts a query names, longest phrasing first so "pan left" is not also read as "pan"."""
    import re
    text = " " + re.sub(r"[^a-z\- ]+", " ", (text or "").lower()) + " "
    found = []
    phrasings = sorted(((w, name) for name, words, _ in CONCEPTS for w in words), key=lambda p: -len(p[0]))
    for words, name in phrasings:
        at = text.find(f" {words} ")
        if at >= 0:
            text = text[:at] + " " * (len(words) + 1) + text[at + len(words) + 1:]
            if name not in found:
                found.append(name)
    return found


def query_scores(text: str, vectors) -> tuple[np.ndarray | None, list[str]]:
    """
    How well each section's movement fits a query: (scores in 0..1, the concepts read).

    The scores are None when the query names no movement, which is how a caller
    tells "a red door" from "slow pan left".
    """
    names = read_query(text)
    if not names:
        return None, []
    V = np.asarray(vectors, np.float32).reshape(-1, len(NAMES))
    by_name = {name: fn for name, _, fn in CONCEPTS}
    parts = [np.asarray(by_name[n](V), np.float32) for n in names]
    return np.mean(parts, axis=0), names

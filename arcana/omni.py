# omni.py -- the shared space: text, pictures, sound and video as comparable vectors
#
# The app's side of the omni encoder. The model runs in a worker process with
# an environment of its own (omni_worker.py says why), started the first time
# something asks for a vector and kept until the app exits, since loading the
# model takes longer than any one query.
#
#     from arcana import omni
#     q = omni.encode_text("rain on a tin roof")           # (768,)
#     v = omni.encode([omni.image("a.jpg"), omni.sound("b.wav"),
#                      omni.clip("c.mp4", 12.0, 22.5)])     # (3, 768)
#
# Every vector is unit length and lives in one space, so a dot product between
# any two is a similarity, whatever kinds they came from.
#
# This module decodes. A picture is read and shrunk here, a video section's
# frames are taken with video.frames_between, sound is read at 16 kHz, and the
# worker receives arrays. See VIDEO_ROADMAP.md, 2.5.

from __future__ import annotations

import atexit
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading

import numpy as np

try:
    from . import modality as _modality
    from . import sections as _sections
    from . import video as _video
except ImportError:                                   # pragma: no cover
    import modality as _modality
    import sections as _sections
    import video as _video

SPACE = "embeddinggemma-2"
MODEL_ID = "google/embeddinggemma-2"
DIM = 768
AUDIO_RATE = 16000
# The model reads 25 tokens of sound a second out of 8,192: a little over five minutes.
AUDIO_SECONDS = 300.0
IMAGE_SIDE = 896
# Frames handed to the model for one section, and their longest side. Measured
# on twelve 8-second sections against six text queries (VIDEO_ROADMAP.md,
# Status): 32 frames at 448 px took 1.24 s a section to encode, 16 at 336 px
# took 0.55 s and kept the top result on five queries of six, with a rank
# correlation of 0.90 to the larger setting.
FRAME_SIDE = 336
FRAMES = 16

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


class OmniError(RuntimeError):
    """The shared-space encoder could not be started, or refused an item."""


def available() -> tuple[bool, str]:
    """Whether the worker can run here, and the reason when it cannot."""
    if getattr(sys, "frozen", False):
        return False, "The installed build does not carry the shared-space encoder yet. Run Arcana from source."
    if shutil.which("uv") is None:
        return False, "The shared-space encoder runs in its own environment, which needs `uv` on the PATH."
    return True, ""


# ─────────────────────────── items ───────────────────────────

def text(value: str, role: str = "query") -> dict:
    return {"kind": "text", "text": value, "role": role}


def image(path: str) -> dict:
    return {"kind": "image", "path": path}


def sound(path: str, t0: float = 0.0, t1: float | None = None) -> dict:
    return {"kind": "audio", "path": path, "t0": t0, "t1": t1}


def clip(path: str, t0: float | None = None, t1: float | None = None) -> dict:
    """A video, or one section of it. A section reference (file#t=a,b) is taken apart here."""
    file, a, b = _modality.split_ref(path)
    return {"kind": "video", "path": file, "t0": a if t0 is None else t0, "t1": b if t1 is None else t1}


def frames(array) -> dict:
    """Frames already decoded, (n, height, width, 3) uint8, as one video item."""
    return {"kind": "video", "frames": np.asarray(array, dtype=np.uint8)}


def item_for(path: str) -> dict:
    """The item for any media file, by its kind."""
    kind = _modality.of_path(path)
    if kind == _modality.AUDIO:
        return sound(path)
    if kind == _modality.VIDEO:
        return clip(path)
    return image(path)


def _array(item: dict) -> np.ndarray:
    """Decode one item to the array the worker takes."""
    kind = item["kind"]
    if kind == "image":
        from PIL import Image
        with Image.open(item["path"]) as im:
            im = im.convert("RGB")
            im.thumbnail((IMAGE_SIDE, IMAGE_SIDE))
            return np.asarray(im, dtype=np.uint8)
    if kind == "video" and item.get("frames") is not None:
        return item["frames"]
    if kind == "video":
        info = _video.probe(item["path"])
        t0 = float(item.get("t0") or 0.0)
        t1 = float(item["t1"]) if item.get("t1") is not None else float(info["duration"])
        t1 = max(t1, t0 + 0.25)
        section = _sections.Section("file", 0, 0, 0, t0, t1, None)
        n = len(_sections.frame_times(section, cap=FRAMES))
        frames = _video.frames_between(item["path"], t0, t1, n, side=FRAME_SIDE, info=info)
        if not frames:
            raise OmniError(f"No frames could be read from {os.path.basename(item['path'])}.")
        return np.stack(frames).astype(np.uint8)
    if kind == "audio":
        t0 = float(item.get("t0") or 0.0)
        t1 = item.get("t1")
        t1 = t0 + AUDIO_SECONDS if t1 is None else min(float(t1), t0 + AUDIO_SECONDS)
        wav = _video.audio(item["path"], AUDIO_RATE, t0=t0, t1=t1)
        if len(wav) < AUDIO_RATE // 10:
            raise OmniError(f"No sound could be read from {os.path.basename(item['path'])}.")
        return wav.astype(np.float32)
    raise OmniError(f"unknown kind {kind!r}")


# ─────────────────────────── the worker ───────────────────────────

class _Worker:
    def __init__(self):
        self.proc = None
        self.lock = threading.Lock()
        self.info: dict = {}
        self.n = 0
        self.tmp = None

    def start(self) -> None:
        ok, why = available()
        if not ok:
            raise OmniError(why)
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "omni_worker.py")
        env = dict(os.environ)
        # The worker is another project as far as uv is concerned: it must not
        # pick up this one's virtual environment or its pins.
        env.pop("VIRTUAL_ENV", None)
        self.tmp = tempfile.mkdtemp(prefix="arcana-omni-")
        self.log = open(os.path.join(self.tmp, "worker.log"), "w", encoding="utf-8", errors="replace")
        self.proc = subprocess.Popen(
            ["uv", "run", "--no-project", "--script", script], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=self.log, text=True, encoding="utf-8", env=env,
            creationflags=_NO_WINDOW, cwd=self.tmp)
        first = self._read()
        if not first.get("ready"):
            self.stop()
            raise OmniError("The shared-space encoder did not start: " + str(first.get("error", "no answer")))
        self.info = first

    def _read(self) -> dict:
        while True:
            line = self.proc.stdout.readline()
            if not line:
                tail = ""
                try:
                    self.log.flush()
                    with open(self.log.name, encoding="utf-8", errors="replace") as fh:
                        tail = fh.read()[-600:]
                except OSError:
                    pass
                return {"ready": False, "ok": False, "error": "the worker stopped. " + tail.strip()}
            line = line.strip()
            if line.startswith("{"):
                try:
                    return json.loads(line)
                except ValueError:
                    continue

    def ask(self, items: list[dict]) -> np.ndarray:
        with self.lock:
            if self.proc is None or self.proc.poll() is not None:
                self.start()
            self.n += 1
            out = os.path.join(self.tmp, f"out_{self.n}.npy")
            sent, files = [], []
            try:
                for k, it in enumerate(items):
                    if it["kind"] == "text":
                        sent.append({"kind": "text", "text": it["text"], "role": it.get("role", "query")})
                        continue
                    path = os.path.join(self.tmp, f"in_{self.n}_{k}.npy")
                    np.save(path, _array(it))
                    files.append(path)
                    sent.append({"kind": it["kind"], "npy": path})
                self.proc.stdin.write(json.dumps({"id": self.n, "items": sent, "out": out}) + "\n")
                self.proc.stdin.flush()
                reply = self._read()
                if not reply.get("ok"):
                    raise OmniError("The shared-space encoder refused: " + str(reply.get("error")))
                return np.load(out)
            finally:
                for f in files + [out]:
                    try:
                        os.remove(f)
                    except OSError:
                        pass

    def stop(self) -> None:
        proc, self.proc = self.proc, None
        if proc is not None:
            try:
                proc.stdin.write('{"quit": true}\n')
                proc.stdin.flush()
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
        try:
            self.log.close()
        except Exception:
            pass
        if self.tmp:
            shutil.rmtree(self.tmp, ignore_errors=True)


_WORKER = _Worker()
atexit.register(_WORKER.stop)


def encode(items: list[dict], batch: int = 8) -> np.ndarray:
    """Vectors for a list of items, one row each, unit length, in the shared space."""
    if not items:
        return np.zeros((0, DIM), np.float32)
    out = [_WORKER.ask(items[i:i + batch]) for i in range(0, len(items), batch)]
    return np.concatenate(out).astype(np.float32)


def encode_text(value: str) -> np.ndarray:
    return encode([text(value)])[0]


def info() -> dict:
    """What the running worker said about itself: model, device, dimension."""
    return dict(_WORKER.info)


def stop() -> None:
    _WORKER.stop()

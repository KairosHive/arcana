# media.py -- what the browser may look at, and how it is drawn
#
# The media routes take a path straight from the query string. Without a
# boundary that is an arbitrary file read, so the server decides what is
# servable from the roots it was configured with plus the files of datasets it
# has loaded, never from what the request asks for. The allowlist, the cached
# thumbnail, preview, waveform, spectrogram and palette builders, and the Flask
# routes that serve them all live here. arcana.py re-exports the names the
# tests pin.
#
#   /thumb?p=          192px JPEG, letterboxed
#   /preview?p=&w=&q=  resized JPEG
#   /palette?p=&n=&w=&h=&db=   a strip of dominant colours
#   /audio?p=          the audio file itself
#   /awave?p=          waveform PNG
#   /aspec?p=          mel spectrogram PNG
#   /board-page/<slug> a board's designed page, sandboxed
#   /film-video?p=     a rendered film, from output/ only
#   /download?p=       a file from output/ as an attachment (board PDFs)

from __future__ import annotations

import hashlib
import os
import pickle
import threading
from functools import lru_cache

import cv2
import numpy as np

try:
    from . import paths as _paths
    from . import gpu as _gpu
    from .cvio import imread_unicode
except ImportError:                                   # pragma: no cover
    import paths as _paths
    import gpu as _gpu
    from cvio import imread_unicode

APP_ROOT = _paths.APP_ROOT
LATENTS_DIR = _paths.subdir("latents")
DB_DIR = _paths.subdir("databases")
OUTPUT_DIR = _paths.subdir("output")
MEDIA_ROOTS = _paths.media_roots()
IMAGES_ROOT = (MEDIA_ROOTS or [os.path.abspath(os.path.join(APP_ROOT, "..", "images"))])[0]
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}

# ---- spectrogram defaults and cache ----
SPEC_CACHE_DIR = os.path.join(_paths.subdir("cache"), "specs")
SPEC_PREVIEW_SR = 16000
SPEC_PREVIEW_SEC = None
SPEC_N_MELS = 96
SPEC_NFFT = 1024
SPEC_HOP = 256
SPEC_WIDTH = 900
SPEC_HEIGHT = 160


def resolve_path(p: str) -> str:
    """A stored path as an absolute one: relative paths live under the images root."""
    return p if os.path.isabs(p) else os.path.join(IMAGES_ROOT, p)


# ─────────────────────────── the allowlist ───────────────────────────

_ALLOWED_ROOTS: set[str] = set()
_ALLOWED_FILES: set[str] = set()
_ALLOWED_ROOTS_LOCK = threading.Lock()
_REGISTERED_DATASETS: set[str] = set()


def _seed_allowed_roots() -> None:
    for r in MEDIA_ROOTS:
        if os.path.isdir(r):
            _ALLOWED_ROOTS.add(os.path.realpath(os.path.abspath(r)))
    if os.path.isdir(OUTPUT_DIR):
        _ALLOWED_ROOTS.add(os.path.realpath(os.path.abspath(OUTPUT_DIR)))


_seed_allowed_roots()


def _is_too_broad(real: str) -> bool:
    """Would allowing this directory as a prefix root expose most of the machine?"""
    if os.path.dirname(real) == real:
        return True
    try:
        home = os.path.realpath(os.path.expanduser("~"))
    except OSError:
        return False
    return real == home or real == os.path.dirname(home)


def register_media_root(path: str) -> None:
    """Mark a directory tree as servable. Prefix access, so only for named folders."""
    if not path:
        return
    try:
        real = os.path.realpath(os.path.abspath(path))
    except OSError:
        return
    if not os.path.isdir(real):
        return
    if _is_too_broad(real):
        print(f"[media] refusing to serve all of {real!r}: too broad to be a media root")
        return
    with _ALLOWED_ROOTS_LOCK:
        _ALLOWED_ROOTS.add(real)


def register_dataset_files(idx2path: dict, cache_key: str | None = None) -> None:
    """Allow exactly the files a dataset names. Not their folders, never a prefix."""
    if cache_key is not None:
        with _ALLOWED_ROOTS_LOCK:
            if cache_key in _REGISTERED_DATASETS:
                return
            _REGISTERED_DATASETS.add(cache_key)
    files = set()
    for v in idx2path.values():
        try:
            files.add(os.path.normcase(os.path.abspath(str(v))))
        except Exception:
            continue
    if files:
        with _ALLOWED_ROOTS_LOCK:
            _ALLOWED_FILES.update(files)


register_dataset_roots = register_dataset_files
register_dataset_dirs = register_dataset_files


def resolve_media_request(p: str) -> str | None:
    """A client-supplied media path as a real one, or None if it is not servable."""
    if not p:
        return None
    full = resolve_path(p)
    try:
        real = os.path.realpath(os.path.abspath(full))
    except OSError:
        return None
    with _ALLOWED_ROOTS_LOCK:
        roots = list(_ALLOWED_ROOTS)
        files = _ALLOWED_FILES
    allowed = _paths.is_within(real, roots)
    if not allowed:
        with _ALLOWED_ROOTS_LOCK:
            allowed = (os.path.normcase(os.path.abspath(full)) in files
                       or os.path.normcase(real) in files)
    if not allowed:
        try:
            from . import ops as _ops
        except ImportError:                           # pragma: no cover
            import ops as _ops
        if _ops.archive.handle_for_path(full) is not None:
            with _ALLOWED_ROOTS_LOCK:
                _ALLOWED_FILES.update({os.path.normcase(os.path.abspath(full)), os.path.normcase(real)})
            allowed = True
    if not allowed or not os.path.isfile(real):
        return None
    return full


def clear_caches() -> None:
    """After a relocation: thumbnails and previews are keyed by path."""
    make_thumbnail_bytes.cache_clear()
    make_resized_bytes.cache_clear()
    make_waveform_png.cache_clear()
    with _ALLOWED_ROOTS_LOCK:
        _REGISTERED_DATASETS.clear()


# ─────────────────────────── cached builders ───────────────────────────

class _MediaUnavailable(Exception):
    """Raised inside a cached builder so a failure is not memoised."""


def _uncached_none(fn):
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except _MediaUnavailable:
            return None
    wrapper.__name__ = getattr(fn, "__name__", "wrapper")
    wrapper.cache_clear = getattr(fn, "cache_clear", lambda: None)
    wrapper.cache_info = getattr(fn, "cache_info", lambda: None)
    return wrapper


@lru_cache(maxsize=50000)
def make_thumbnail_bytes(path: str, max_side: int = 192) -> bytes | None:
    img = imread_unicode(resolve_path(path))
    if img is None:
        raise _MediaUnavailable(path)
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    pad = np.zeros((max_side, max_side, 3), dtype=np.uint8)
    x0, y0 = (max_side - new_w) // 2, (max_side - new_h) // 2
    pad[y0:y0 + new_h, x0:x0 + new_w] = img
    ok, buf = cv2.imencode(".jpg", pad, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return buf.tobytes() if ok else None


make_thumbnail_bytes = _uncached_none(make_thumbnail_bytes)


@lru_cache(maxsize=2000)
def make_resized_bytes(path: str, max_w: int = 900, q: int = 72) -> bytes | None:
    img = imread_unicode(resolve_path(path))
    if img is None:
        raise _MediaUnavailable(path)
    h, w = img.shape[:2]
    if w > max_w:
        img = cv2.resize(img, (max_w, int(h * (max_w / float(w)))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    return buf.tobytes() if ok else None


make_resized_bytes = _uncached_none(make_resized_bytes)


def read_audio_mono(path, target_sr=24000, seconds=None, pad=False):
    """One decoder for the app and the indexer, imported late so start-up stays light."""
    try:
        from .db import read_audio_mono as _decode
    except ImportError:                               # pragma: no cover
        from db import read_audio_mono as _decode
    return _decode(resolve_path(path), target_sr=target_sr, seconds=seconds, pad=pad)


@lru_cache(maxsize=20000)
def make_waveform_png(path: str, width=900, height=160) -> bytes | None:
    try:
        x, _ = read_audio_mono(path, target_sr=24000, seconds=None, pad=False)
        x = x / (np.max(np.abs(x)) + 1e-8)
        img = np.full((height, width, 3), 18, dtype=np.uint8)
        mid = height // 2
        xs = np.linspace(0, len(x) - 1, width).astype(int)
        ys = (x[xs] * (height * 0.45)).astype(int)
        for i in range(1, width):
            cv2.line(img, (i - 1, int(mid - ys[i - 1])), (i, int(mid - ys[i])), (200, 200, 200), 1)
        ok, buf = cv2.imencode(".png", img)
        return buf.tobytes() if ok else None
    except Exception as e:
        print("[waveform] error:", e)
        raise _MediaUnavailable(path)


make_waveform_png = _uncached_none(make_waveform_png)

_SPEC_LUT = None


def _lut():
    global _SPEC_LUT
    if _SPEC_LUT is None:
        from matplotlib import pyplot as plt
        _SPEC_LUT = (plt.get_cmap("magma")(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
    return _SPEC_LUT


def _spec_cache_key(full_path: str, mtime: float, params: tuple) -> str:
    s = f"{full_path}|{mtime}|" + "|".join(map(str, params)) + "|v3"
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def _mel_db(y: np.ndarray, sr: int, n_fft: int, hop: int, n_mels: int) -> np.ndarray:
    try:
        import torch
        import torchaudio
    except Exception:
        torchaudio = None
    if torchaudio is not None:
        device = _gpu.device()
        wav = torch.from_numpy(y).to(device).unsqueeze(0)
        mel = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=n_fft, hop_length=hop,
                                                   n_mels=n_mels, power=2.0).to(device)(wav)
        db = torchaudio.transforms.AmplitudeToDB(stype="power").to(device)(mel)
        return db.squeeze(0).detach().cpu().numpy()
    import librosa
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels, power=2.0)
    return librosa.power_to_db(S, ref=np.max)


def make_melspec_png(path: str, width_px: int = SPEC_WIDTH, height_px: int = SPEC_HEIGHT,
                     sr: int = SPEC_PREVIEW_SR, seconds=SPEC_PREVIEW_SEC, n_fft: int = SPEC_NFFT,
                     hop_length: int = SPEC_HOP, n_mels: int = SPEC_N_MELS) -> bytes | None:
    try:
        full = resolve_path(path)
        if not os.path.exists(full):
            return None
        key = _spec_cache_key(full, os.path.getmtime(full),
                              (width_px, height_px, sr, seconds, n_fft, hop_length, n_mels))
        cache_file = os.path.join(SPEC_CACHE_DIR, f"{key}.png")
        try:
            with open(cache_file, "rb") as fh:
                return fh.read()
        except OSError:
            pass
        y, _ = read_audio_mono(path, target_sr=sr, seconds=seconds, pad=False)
        S_db = _mel_db(y, sr, n_fft, hop_length, n_mels)
        S_db = np.nan_to_num(S_db, nan=np.min(S_db))
        mn, mx = float(S_db.min()), float(S_db.max())
        rng = (mx - mn) if (mx > mn) else 1.0
        img8 = np.flipud(((S_db - mn) / rng * 255.0).astype(np.uint8))
        img8 = cv2.resize(img8, (width_px, height_px), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".png", _lut()[img8])
        if not ok:
            return None
        data = buf.tobytes()
        try:
            _paths.ensure_dir(SPEC_CACHE_DIR)
            with open(cache_file, "wb") as fh:
                fh.write(data)
        except OSError as e:
            print("[spec-cache] write failed:", e)
        return data
    except Exception as e:
        print("[melspec] error:", e)
        return None


# ─────────────────────────── palette cache ───────────────────────────

_palette_cache: dict = {}
_palette_cache_lock = threading.Lock()


def _load_palette_cache(db_name: str) -> dict:
    palette_path = os.path.join(DB_DIR, f"features_{db_name}_palette.npz")
    index_path = os.path.join(DB_DIR, f"index_{db_name}_image.pkl")
    if not (os.path.exists(palette_path) and os.path.exists(index_path)):
        return {}
    try:
        data = np.load(palette_path)
        ids, dominant = data["ids"], data["dominant"]
        with open(index_path, "rb") as f:
            _blob, idx2path = pickle.load(f)
        idx2path = {int(k): str(v) for k, v in idx2path.items()}
        cache = {}
        for i, img_id in enumerate(ids):
            path = idx2path.get(int(img_id))
            if path is not None:
                cache[os.path.normpath(path)] = dominant[i]
        return cache
    except Exception as e:
        print(f"[palette cache] Failed to load {palette_path}: {e}")
        return {}


def preload_palette_cache(db_name: str) -> None:
    with _palette_cache_lock:
        if db_name not in _palette_cache:
            _palette_cache[db_name] = _load_palette_cache(db_name)


def get_cached_palette(path: str, db_name: str | None = None) -> np.ndarray | None:
    norm_path = os.path.normpath(path)
    with _palette_cache_lock:
        for cache in _palette_cache.values():
            if norm_path in cache:
                return cache[norm_path]
        if db_name and db_name not in _palette_cache:
            _palette_cache[db_name] = _load_palette_cache(db_name)
            if norm_path in _palette_cache[db_name]:
                return _palette_cache[db_name][norm_path]
    return None


def palette_swatch_png(path: str, n_colors: int, width: int, height: int, db_name: str | None) -> bytes:
    """A horizontal strip of the picture's dominant colours, widths by share."""
    try:
        cached = get_cached_palette(path, db_name)
        if cached is not None:
            palette = cached[:n_colors]
            colors, proportions = palette[:, :3], palette[:, 3]
            proportions = proportions / (proportions.sum() + 1e-8)
        else:
            try:
                from .palette import extract_dominant_colors
            except ImportError:                       # pragma: no cover
                from palette import extract_dominant_colors
            palette = extract_dominant_colors(path, n_colors=n_colors)
            colors, proportions = palette[:, :3], palette[:, 3]
        n = len(colors)
        lab = np.zeros((n, 1, 3), dtype=np.float32)
        lab[:, 0, :] = colors
        lab[:, :, 0] *= 255.0 / 100.0
        lab[:, :, 1] += 128.0
        lab[:, :, 2] += 128.0
        bgr = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
        swatch = np.zeros((height, width, 3), dtype=np.uint8)
        x = 0
        for i, prop in enumerate(proportions):
            w = int(prop * width)
            if i == len(proportions) - 1:
                w = width - x
            if w > 0:
                swatch[:, x:x + w] = bgr[i, 0]
                x += w
        ok, buf = cv2.imencode(".png", swatch)
        if ok:
            return buf.tobytes()
    except Exception as e:
        print(f"[palette endpoint] Error: {e}")
    ok, buf = cv2.imencode(".png", np.full((height, width, 3), 64, dtype=np.uint8))
    return buf.tobytes()


# ─────────────────────────── the routes ───────────────────────────

def blueprint():
    from flask import Blueprint, Response, request, send_file

    bp = Blueprint("arcana_media", __name__)
    YEAR = {"Cache-Control": "public, max-age=31536000"}

    def _path():
        p = request.args.get("p")
        if not p:
            return None, Response("missing p", status=400)
        path = resolve_media_request(p)
        if path is None:
            return None, Response(status=404)
        return path, None

    def _int(name, default, lo, hi):
        try:
            return min(max(lo, int(request.args.get(name, default))), hi)
        except Exception:
            return default

    @bp.get("/thumb")
    def _thumb():
        path, err = _path()
        if err:
            return err
        data = make_thumbnail_bytes(path)
        return Response(data, mimetype="image/jpeg", headers=YEAR) if data else Response(status=404)

    @bp.get("/preview")
    def _preview():
        path, err = _path()
        if err:
            return err
        data = make_resized_bytes(path, max_w=_int("w", 900, 200, 2048), q=_int("q", 72, 50, 95))
        return Response(data, mimetype="image/jpeg", headers=YEAR) if data else Response(status=404)

    @bp.get("/palette")
    def _palette():
        path, err = _path()
        if err:
            return err
        data = palette_swatch_png(path, _int("n", 16, 4, 32), _int("w", 300, 100, 600),
                                  _int("h", 30, 20, 60), request.args.get("db") or None)
        return Response(data, mimetype="image/png", headers={"Cache-Control": "public, max-age=86400"})

    @bp.get("/audio")
    def _audio():
        path, err = _path()
        if err:
            return err
        full = resolve_path(path)
        ext = os.path.splitext(full)[1].lower()
        mime = {".mp3": "audio/mpeg", ".m4a": "audio/mp4", ".aac": "audio/aac",
                ".flac": "audio/flac", ".ogg": "audio/ogg"}.get(ext, "audio/wav")
        return send_file(full, mimetype=mime, conditional=True, max_age=31536000)

    @bp.get("/awave")
    def _awave():
        path, err = _path()
        if err:
            return err
        data = make_waveform_png(path)
        return Response(data, mimetype="image/png", headers=YEAR) if data else Response(status=404)

    @bp.get("/aspec")
    def _aspec():
        path, err = _path()
        if err:
            return err
        data = make_melspec_png(path)
        return Response(data, mimetype="image/png", headers=YEAR) if data else Response(status=404)

    @bp.get("/board-page/<slug>")
    def _board_page(slug):
        try:
            from . import ops as _ops
        except ImportError:                           # pragma: no cover
            import ops as _ops
        path = _ops.board.page_file(slug)
        if not path:
            return Response("This board has no page.", status=404)
        with open(path, "rb") as fh:
            data = fh.read()
        if request.args.get("print"):
            hook = (b"<script>addEventListener('load',function(){"
                    b"setTimeout(function(){print()},500)})</script>")
            at = data.rfind(b"</body>")
            data = data[:at] + hook + data[at:] if at >= 0 else data + hook
        resp = Response(data, mimetype="text/html")
        resp.headers["Content-Security-Policy"] = ("sandbox allow-scripts allow-modals; "
                                                   "connect-src 'none'; form-action 'none'")
        resp.headers["Cache-Control"] = "no-store"
        return resp

    def _output_file(p: str, suffixes: tuple[str, ...]) -> str | None:
        real = os.path.realpath(p or "")
        if not (p and p.lower().endswith(suffixes) and os.path.isfile(real)
                and _paths.is_within(real, [os.path.realpath(OUTPUT_DIR)])):
            return None
        return real

    @bp.get("/film-video")
    def _film_video():
        real = _output_file(request.args.get("p") or "", (".mp4",))
        if real is None:
            return Response(status=404)
        return send_file(real, mimetype="video/mp4", conditional=True)

    @bp.get("/download")
    def _download():
        """A file Arcana made, as an attachment: the board PDF, a rendered page."""
        real = _output_file(request.args.get("p") or "", (".pdf", ".html", ".png", ".jpg", ".mp4", ".cube"))
        if real is None:
            return Response(status=404)
        return send_file(real, as_attachment=True, download_name=os.path.basename(real))

    return bp


def register(flask_app) -> None:
    if "arcana_media" not in flask_app.blueprints:
        flask_app.register_blueprint(blueprint())

import dash
from dash import dcc, html, Input, Output, State, ctx
import plotly.express as px
import pandas as pd
import cv2
try:
    from .cvio import imread_unicode, imwrite_unicode
except ImportError:
    from cvio import imread_unicode, imwrite_unicode
import base64
import os
import pickle
import torch
import threading
import json
from usearch.index import Index
from concurrent.futures import ThreadPoolExecutor
import dash_daq as daq
import numpy as np
import plotly.graph_objects as go
import re
from PIL import Image
# diffusers is imported lazily inside the story-mode callback: it costs ~3 s at
# import time and is only needed if someone actually generates images.
from io import BytesIO
from functools import lru_cache
from flask import Response, request
import urllib.parse
from dash import ALL
import hashlib
import shutil  # <— add at top of file with other imports if not present
import tempfile
import librosa
from matplotlib import pyplot as plt

try:
    from . import ui_datasets as _ui_datasets
    from . import ui_style as _ui
    from . import jobs
    from . import boards as _boards
except ImportError:
    import ui_datasets as _ui_datasets
    import ui_style as _ui
    import jobs
    import boards as _boards

# --- palette/style search ---
try:
    from .db import search_by_palette, search_by_style, search_combined, load_palette_features, load_style_features
    PALETTE_STYLE_AVAILABLE = True
except ImportError:
    try:
        from db import search_by_palette, search_by_style, search_combined, load_palette_features, load_style_features
        PALETTE_STYLE_AVAILABLE = True
    except ImportError:
        PALETTE_STYLE_AVAILABLE = False
        search_by_palette = search_by_style = search_combined = None
        load_palette_features = load_style_features = None

# --- color transfer (ModFlows) ---
try:
    from .color_transfer import transfer_colors, get_device_info
    COLOR_TRANSFER_AVAILABLE = True
except ImportError:
    try:
        from color_transfer import transfer_colors, get_device_info
        COLOR_TRANSFER_AVAILABLE = True
    except ImportError:
        COLOR_TRANSFER_AVAILABLE = False
        transfer_colors = get_device_info = None

# --- LAB color transfer (Reinhard) ---
try:
    from .lab_transfer import lab_color_transfer_pil
    LAB_TRANSFER_AVAILABLE = True
except ImportError:
    try:
        from lab_transfer import lab_color_transfer_pil
        LAB_TRANSFER_AVAILABLE = True
    except ImportError:
        LAB_TRANSFER_AVAILABLE = False
        lab_color_transfer_pil = None





# --- audio + CLAP ---
import soundfile as sf
try:
    import torchaudio
except Exception:
    torchaudio = None



torch.set_grad_enabled(False)

try:
    from . import paths as _paths
except ImportError:
    import paths as _paths

try:
    from . import gpu as _gpu
except ImportError:
    import gpu as _gpu

# Every writable location comes from paths.py, which keeps data outside the
# install directory. Nothing here creates a directory: a packaged app installs
# read-only, so an os.makedirs at import time is a crash on launch.
APP_ROOT = _paths.APP_ROOT
LATENTS_DIR = _paths.subdir("latents")
DB_DIR = _paths.subdir("databases")
BUNDLES_DIR = _paths.subdir("bundles")

MEDIA_ROOTS = _paths.media_roots()
IMAGES_ROOT = (MEDIA_ROOTS or [os.path.abspath(os.path.join(APP_ROOT, "..", "images"))])[0]

OUTPUT_DIR = _paths.subdir("output")
STORIES_DIR = os.path.join(OUTPUT_DIR, "stories")
SELECTIONS_DIR = os.path.join(OUTPUT_DIR, "selections")
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}

# ---- FAST SPEC CONFIG + CACHE ----
SPEC_CACHE_DIR = os.path.join(_paths.subdir("cache"), "specs")

# small but good-looking preview defaults (tweak if you want)
SPEC_PREVIEW_SR = 16000
SPEC_PREVIEW_SEC = None
SPEC_N_MELS = 96
SPEC_NFFT = 1024
SPEC_HOP = 256
SPEC_WIDTH = 900
SPEC_HEIGHT = 160

# 256-color look-up table (no heavy plotting)
_SPEC_LUT = (plt.get_cmap("magma")(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)

# lightweight threadpool to prewarm cache
SPEC_EXEC = ThreadPoolExecutor(max_workers=max(2, (os.cpu_count() or 4)//2))



# (output directories are created on first save, not at import -- see save_images)


# ------------- FILE DISCOVERY HELPERS -------------

from functools import lru_cache

def _slugify(txt: str, maxlen: int = 40) -> str:
    s = "".join(c if c.isalnum() else "-" for c in txt).strip("-")
    s = re.sub(r"-+", "-", s)
    return s[:maxlen] or "text"

def _short_poetry_name(img_path: str, prompt: str, idx: int, ext: str = "png") -> str:
    base_hash = hashlib.md5(img_path.encode("utf-8")).hexdigest()[:8]
    prompt_slug = _slugify(prompt, maxlen=36)
    return f"{idx:02d}_{base_hash}_{prompt_slug}_poetry.{ext}"

def _win_longpath(p: str) -> str:
    # Optional: makes Windows accept very long absolute paths.
    if os.name == "nt":
        ap = os.path.abspath(p)
        if not ap.startswith("\\\\?\\"):
            return "\\\\?\\" + ap
    return p


class _MediaUnavailable(Exception):
    """Raised inside the cached media builders so a failure is NOT memoised.

    functools.lru_cache stores return values but never exceptions, so raising
    here (and translating back to None at the boundary) keeps successes cached
    while letting a file that was briefly unreadable -- half-written, on a
    network drive, still uploading -- succeed on the next request instead of
    404ing for the rest of the session.
    """


def _uncached_none(fn):
    """Adapt a cached builder that raises _MediaUnavailable back to returning None."""
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
    full_path = resolve_path(path)
    img = imread_unicode(full_path)
    if img is None:
        raise _MediaUnavailable(path)
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    pad = np.zeros((max_side, max_side, 3), dtype=np.uint8)
    x0 = (max_side - new_w) // 2
    y0 = (max_side - new_h) // 2
    pad[y0 : y0 + new_h, x0 : x0 + new_w] = img
    ok, buf = cv2.imencode(".jpg", pad, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return buf.tobytes() if ok else None


from functools import lru_cache


make_thumbnail_bytes = _uncached_none(make_thumbnail_bytes)


@lru_cache(maxsize=2000)  # cache a few thousand medium previews
def make_resized_bytes(path: str, max_w: int = 900, q: int = 72) -> bytes | None:
    full_path = resolve_path(path)
    img = imread_unicode(full_path)
    if img is None:
        raise _MediaUnavailable(path)
    h, w = img.shape[:2]
    if w > max_w:
        new_w = max_w
        new_h = int(h * (max_w / float(w)))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    return buf.tobytes() if ok else None


make_resized_bytes = _uncached_none(make_resized_bytes)


@lru_cache(maxsize=50000)  # 13k fits
def thumb_b64_for(path: str) -> str | None:
    return encode_thumbnail(path)  # uses resolve_path inside


def read_audio_mono(path, target_sr=24000, seconds=None, pad=False):
    """
    Decode an audio file to mono float32 at target_sr.

    This used to be a second, independent copy of db.read_audio_mono. The two
    drifted: db's was fixed to try soundfile before torchaudio (torchaudio 2.9
    dropped its own decoding backends and now raises ImportError unless
    TorchCodec is installed), and this one was not -- so indexing audio worked
    while every waveform and spectrogram in the UI 404'd. There is now one
    implementation, and the import is deferred so app start-up does not pay for
    db's model imports.
    """
    try:
        from .db import read_audio_mono as _decode
    except ImportError:                      # running as a loose script
        from db import read_audio_mono as _decode
    return _decode(resolve_path(path), target_sr=target_sr,
                   seconds=seconds, pad=pad)


@lru_cache(maxsize=20000)
def make_waveform_png(path: str, width=900, height=160) -> bytes | None:
    try:
        x, _ = read_audio_mono(path, target_sr=24000, seconds=None, pad=False)
        x = x / (np.max(np.abs(x)) + 1e-8)
        img = np.full((height, width, 3), 18, dtype=np.uint8)  # dark bg
        mid = height // 2
        xs = np.linspace(0, len(x)-1, width).astype(int)
        ys = (x[xs] * (height*0.45)).astype(int)
        for i in range(1, width):
            y0, y1 = int(mid - ys[i-1]), int(mid - ys[i])
            cv2.line(img, (i-1, y0), (i, y1), (200,200,200), 1)
        ok, buf = cv2.imencode(".png", img)
        return buf.tobytes() if ok else None
    except Exception as e:
        print("[waveform] error:", e)
        raise _MediaUnavailable(path)


make_waveform_png = _uncached_none(make_waveform_png)


def _cosine_group(keys, paths, index: Index, thresh: float = 0.08):
    """
    Group the top-N results by cosine distance threshold (1 - cosine_sim).
    Returns: [{'gid': 'g0', 'keys': [...], 'paths': [...]}...], preserving rank order.
    """
    if len(keys) <= 1:
        return [{"gid": "g0", "keys": keys, "paths": paths}]

    vecs = np.stack([index.get(k) for k in keys]).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
    sim = vecs @ vecs.T
    dist = 1.0 - sim

    parent = list(range(len(keys)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    n = len(keys)
    for i in range(n):
        for j in range(i + 1, n):
            if dist[i, j] <= thresh:
                union(i, j)

    buckets = {}
    for i in range(n):
        r = find(i)
        buckets.setdefault(r, []).append(i)

    ordered = sorted(buckets.values(), key=lambda idxs: min(idxs))
    groups = []
    for gi, idxs in enumerate(ordered):
        groups.append(
            {
                "gid": f"g{gi}",
                "keys": [keys[i] for i in idxs],
                "paths": [paths[i] for i in idxs],
            }
        )
    return groups


def get_latent_options(latent_dir=LATENTS_DIR, n_dim=2):
    pattern = re.compile(rf"latent_space_(.+)_{n_dim}d\.pkl$")
    options = []
    for fname in _paths.listdir_safe(latent_dir):
        m = pattern.match(fname)
        if m:
            options.append({"label": m.group(1), "value": m.group(1)})
    return sorted(options, key=lambda x: x["label"])


def resolve_path(p):
    # If the stored path is absolute, use it; else fall back to IMAGES_ROOT
    return p if os.path.isabs(p) else os.path.join(IMAGES_ROOT, p)


def _parse_dataset_value(dataset_value: str):
    """Split "<name>::<dim>::<modality>" (older values omit the modality)."""
    parts = (dataset_value or "").split("::")
    if len(parts) == 3:
        return parts[0], int(parts[1]), parts[2]
    return parts[0], int(parts[1]), "image"


def _safe_output_dir(*parts) -> str:
    """
    Build a path under OUTPUT_DIR from components, some of which come from a
    text box the user types into. Anything that would escape OUTPUT_DIR --
    "../../..", an absolute path, a drive letter -- is rejected rather than
    silently writing outside the output tree. The directory is created here,
    which is also the only place output directories get created at all.
    """
    cleaned = [str(p) for p in parts if p not in (None, "")]
    if not cleaned:
        cleaned = ["untitled"]
    target = _paths.safe_join(OUTPUT_DIR, *cleaned)
    if target is None:
        raise ValueError(
            "That folder name is not allowed - use a plain name without "
            "slashes, drive letters, or '..'."
        )
    return _paths.ensure_dir(target)


# ------------- SERVABLE MEDIA ALLOWLIST -------------
# The media endpoints take a path straight from the query string. Without a
# boundary that is an arbitrary file read: /audio?p=C:\...\secrets.env returns
# the file. The server therefore decides what is servable, from the roots it has
# been configured with plus the roots of datasets it has actually loaded --
# never from what the request asks for.
_ALLOWED_ROOTS: set[str] = set()
# The exact files a loaded dataset names. Membership is the grant -- not the
# folder they sit in, and never a prefix. See register_dataset_files().
_ALLOWED_FILES: set[str] = set()
_ALLOWED_ROOTS_LOCK = threading.Lock()


def _seed_allowed_roots() -> None:
    for r in MEDIA_ROOTS:
        if os.path.isdir(r):
            _ALLOWED_ROOTS.add(os.path.realpath(os.path.abspath(r)))
    # Arcana's own output tree: uploaded reference images, colour-transfer
    # results, saved selections. Produced by the app, so serving it back is not
    # arbitrary filesystem access. It may not exist yet on a fresh install,
    # which is why writers also call register_media_root() after creating it.
    if os.path.isdir(OUTPUT_DIR):
        _ALLOWED_ROOTS.add(os.path.realpath(os.path.abspath(OUTPUT_DIR)))


_seed_allowed_roots()


def _is_too_broad(real: str) -> bool:
    """
    Would allowing this directory as a PREFIX root expose most of the machine?

    A drive root, the filesystem root, the user's home, or the directory holding
    all users are never acceptable as prefix roots: everything the user owns
    lives under them.
    """
    if os.path.dirname(real) == real:          # C:\  or  /
        return True
    try:
        home = os.path.realpath(os.path.expanduser("~"))
    except OSError:
        return False
    if real == home or real == os.path.dirname(home):   # ~ , C:\Users , /home
        return True
    return False


def register_media_root(path: str) -> None:
    """
    Mark a directory tree as servable.

    This grants PREFIX access -- everything beneath `path` becomes readable --
    so it is only for directories the user or the configuration named directly
    (the images/ folder, the app's own output tree, a relocation target). Paths
    derived from dataset contents must go through register_dataset_dirs().
    """
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


_REGISTERED_DATASETS: set[str] = set()


def register_dataset_files(idx2path: dict, cache_key: str | None = None) -> None:
    """
    Allow exactly the files a dataset names. Not their folders. Never a prefix.

    This used to register os.path.commonpath() of the containing directories as
    a prefix root, which is a whole-drive arbitrary file read the moment a
    dataset spans two top-level folders: [C:\\media\\img\\a.png, C:\\scratch.png]
    collapses to 'C:\\' and /audio then returns any file on the drive. Two
    folders under the home directory collapse to the home directory, which is
    just as bad. Granting the parent directories instead is narrower, but still
    hands over every sibling of every indexed file.

    So the grant is the file itself: a dataset can only ever expose what it
    actually indexes.

    `cache_key` skips the work for a dataset already seen -- this runs from
    load_data(), called on every scatter update, and the largest dataset has 82k
    paths.
    """
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


# Older call sites named these "roots"/"dirs"; they were always dataset file lists.
register_dataset_roots = register_dataset_files
register_dataset_dirs = register_dataset_files


def resolve_media_request(p: str) -> str | None:
    """
    Turn a client-supplied media path into a real path, or None if this server
    is not willing to serve it.

    Two independent grants, deliberately different in shape:
      - _ALLOWED_ROOTS  prefix match, for directories named by the user or config
      - _ALLOWED_FILES  exact file match, for the contents of a loaded dataset
    """
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
        return None
    if not os.path.isfile(real):
        return None
    return full


def get_db_options(db_dir=DB_DIR):
    pattern = re.compile(r"index_(.+)\.pkl$")
    options = []
    for fname in _paths.listdir_safe(db_dir):
        m = pattern.match(fname)
        if m:
            options.append({"label": m.group(1), "value": m.group(1)})
    return sorted(options, key=lambda x: x["label"])


def get_matching_datasets(latent_dir=LATENTS_DIR, db_dir=DB_DIR):
    # Files: latent_space_<name>_<mod>_<dim>d.pkl  &  index_<name>_<mod>.pkl
    lat_pat = re.compile(r"latent_space_(.+)_(image|audio)_(\d+)d\.pkl$")
    db_pat  = re.compile(r"index_(.+)_(image|audio)\.pkl$")
    lat_map = {}
    for fname in _paths.listdir_safe(latent_dir):
        m = lat_pat.match(fname)
        if m:
            name, mod, dim = m.group(1), m.group(2), m.group(3)
            lat_map.setdefault((name, mod), []).append(dim)
    db_keys = {(m.group(1), m.group(2)) for fname in _paths.listdir_safe(db_dir) if (m := db_pat.match(fname))}
    options = []
    for (name, mod), dims in lat_map.items():
        if (name, mod) in db_keys:
            for d in sorted(dims):
                options.append({
                    "label": f"{name} · {mod} ({d}D)",
                    "value": f"{name}::{d}::{mod}"
                })
    return sorted(options, key=lambda x: x["label"])



dataset_options = get_matching_datasets()
default_dataset = dataset_options[0]["value"] if dataset_options else None


# ------------- DATA LOADING HELPERS -------------
def encode_image(image_path, max_width=1024):
    full_path = resolve_path(image_path)  # CHANGED
    image = imread_unicode(full_path)
    if image is None:
        print(f"[ERROR] Could not load image: {full_path}")
        return None
    h, w = image.shape[:2]
    if w > max_width:
        scale = max_width / float(w)
        new_w, new_h = int(w * scale), int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    _, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
    return base64.b64encode(buffer).decode()


def encode_thumbnail(path, max_side=128):
    full_path = resolve_path(path)  # CHANGED
    img = imread_unicode(full_path)
    if img is None:
        print(f"[ERROR] Could not load image: {full_path}")
        return None
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    thumb = np.zeros((max_side, max_side, 3), dtype=np.uint8)
    x_offset = (max_side - new_w) // 2
    y_offset = (max_side - new_h) // 2
    thumb[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = img
    _, buffer = cv2.imencode(".jpg", thumb, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    img_str = base64.b64encode(buffer).decode()
    return f"data:image/jpeg;base64,{img_str}"


def load_data(name, n_dim=2, modality="image"):
    latent_path = os.path.join(LATENTS_DIR, f"latent_space_{name}_{modality}_{n_dim}d.pkl")
    df = pd.read_pickle(latent_path)

    if "label" in df.columns:
        df["label"] = df["label"].astype(str)
    if "path" in df.columns:
        df["path"] = df["path"].astype(str)
        # Thumbnails are requested straight from this frame, before any search
        # has loaded the index -- so the dataset's directories have to become
        # servable here too, or a dataset stored outside the media roots renders
        # as a grid of 404s until you happen to run a prompt search.
        register_dataset_roots(dict(enumerate(df["path"].tolist())),
                               cache_key=f"latent:{name}:{modality}")
    for col in ("x", "y", "z"):
        if col in df.columns:
            df[col] = df[col].astype("float32")
    return df.reset_index(drop=True)

def load_index(name, modality="image"):
    index_name = os.path.join(DB_DIR, f"index_{name}_{modality}.pkl")
    with open(index_name, "rb") as f:
        idx_blob, idx2path = pickle.load(f)
    register_dataset_roots(idx2path, cache_key=f"index:{name}:{modality}")
    return Index.restore(idx_blob), idx2path



# ------------- CLIP TEXT ENCODER (lazy) -------------
# Searching only ever encodes the prompt, so only the text tower is needed:
# 354M parameters instead of the full model's 986M, and ~0.9 s to load instead
# of ~25 s. Loading it lazily also means the app starts (and can show an error)
# without a 4 GB download having to succeed first.
CLIP_MODEL_ID = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"   # fallback only

# Keyed by model id. A single-slot cache was wrong for the same reason it was
# wrong in db.load_clip: the first dataset you opened decided which text
# encoder every later search used.
_CLIP_TEXT: dict = {}
_MODEL_LOAD_LOCK = threading.Lock()


def text_model_for_dim(ndim: int) -> str:
    """
    Which encoder produced vectors of this width.

    A prompt has to be encoded by the same model that encoded the pictures, or
    the two live in unrelated spaces. Nothing in a legacy index records which
    model built it -- but the vector width does, because Arcana's image
    encoders have distinct dimensions (512 / 768 / 1024). Reading it off the
    index means every dataset searches correctly with no migration and no extra
    metadata.
    """
    try:
        from . import models as _models
    except ImportError:
        import models as _models
    for m in _models.MODELS:
        if m.modality == "image" and m.dim == ndim:
            return m.id
    raise RuntimeError(
        f"This dataset's vectors are {ndim}-dimensional, which does not match "
        f"any encoder Arcana knows about. It was probably built by a different "
        f"version; re-index the folder to search it."
    )


def load_clip_text(device: str = "cpu", model_id: str | None = None):
    """Load a CLIP text tower on first use. Raises with something actionable."""
    model_id = model_id or CLIP_MODEL_ID
    with _MODEL_LOAD_LOCK:
        if model_id not in _CLIP_TEXT:
            try:
                from transformers import CLIPTextModelWithProjection, CLIPTokenizerFast
                m = CLIPTextModelWithProjection.from_pretrained(model_id, dtype=torch.float32)
                m.eval().to(device)
                tok = CLIPTokenizerFast.from_pretrained(model_id)
            except Exception as e:
                raise RuntimeError(
                    f"Could not load the text encoder '{model_id}': {e}. "
                    f"The first run needs to download it, so check your internet connection. "
                    f"Set HF_HOME to choose where models are cached."
                ) from e
            _CLIP_TEXT[model_id] = (m, tok)
    return _CLIP_TEXT[model_id]

# --- lazy CLAP (keep FP32 to avoid BN/dtype issues) ---
_CLAP = {"model": None, "proc": None}
def load_clap(device="cpu"):
    with _MODEL_LOAD_LOCK:
        if _CLAP["model"] is None:
            from transformers import ClapModel, ClapProcessor
            m = ClapModel.from_pretrained("laion/clap-htsat-fused")
            m.eval().to(device)
            p = ClapProcessor.from_pretrained("laion/clap-htsat-fused")
            _CLAP.update(model=m, proc=p)
    return _CLAP["model"], _CLAP["proc"]


def search(index, idx2path, query, n, modality="image"):
    if modality == "image":
        # Match the encoder to the index. This used to always load ViT-H/14
        # while an index built with ViT-B/32 holds 512-d vectors, so every
        # search against such a dataset died inside usearch with "The number of
        # vector dimensions doesn't match!" and the results panel simply stayed
        # empty. ViT-B/32 is the encoder the panel recommends to anyone without
        # a GPU, so prompt search was broken for exactly the people most likely
        # to choose it.
        clip_model, clip_tok = load_clip_text(
            model_id=text_model_for_dim(int(index.ndim)))
        inputs = clip_tok(query, return_tensors="pt")
        vec = clip_model(**inputs).text_embeds.detach().cpu().numpy().flatten()
    else:
        clap_model, clap_proc = load_clap(device="cpu")
        inputs = clap_proc(text=[query], return_tensors="pt", padding=True)
        # keep FP32
        for k in inputs:
            inputs[k] = inputs[k].to(clap_model.device)
        with torch.no_grad():
            try:
                emb = clap_model.get_text_features(**inputs)
            except AttributeError:
                emb = clap_model(**inputs).text_embeds
        vec = emb.squeeze().detach().cpu().numpy().flatten()

    idxs = index.search(vec, n, exact=True)
    return [(idx.key, idx2path[idx.key], idx.distance) for idx in idxs]



# ------------- PALETTE CACHE -------------
# Cache precomputed dominant colors to avoid K-means on every request
_palette_cache = {}  # db_name -> {path: dominant_colors array}
_palette_cache_lock = threading.Lock()

def _load_palette_cache(db_name: str) -> dict:
    """Load precomputed dominant colors from features file."""
    palette_path = os.path.join(DB_DIR, f"features_{db_name}_palette.npz")
    if not os.path.exists(palette_path):
        return {}
    
    try:
        data = np.load(palette_path)
        ids = data['ids']
        dominant = data['dominant']  # (n_images, n_colors, 4)
        
        # Build path -> dominant mapping. This used to read
        # databases/<name>_image.json, which db.py has never written -- so the
        # cache was always empty and /palette re-ran k-means on every request
        # (~2.5 s per uncached image). The mapping it needs is the idx2path that
        # already sits in the index pickle.
        index_path = os.path.join(DB_DIR, f"index_{db_name}_image.pkl")
        if not os.path.exists(index_path):
            return {}

        with open(index_path, "rb") as f:
            _blob, idx2path = pickle.load(f)
        idx2path = {int(k): str(v) for k, v in idx2path.items()}

        cache = {}
        for i, img_id in enumerate(ids):
            path = idx2path.get(int(img_id))
            if path is not None:
                # Normalize path for consistent lookup
                cache[os.path.normpath(path)] = dominant[i]
        return cache
    except Exception as e:
        print(f"[palette cache] Failed to load {palette_path}: {e}")
        return {}

def get_cached_palette(path: str, db_name: str = None) -> np.ndarray | None:
    """Get precomputed dominant colors for a path."""
    norm_path = os.path.normpath(path)
    
    # Try all loaded caches
    with _palette_cache_lock:
        for name, cache in _palette_cache.items():
            if norm_path in cache:
                return cache[norm_path]
        
        # If db_name specified and not loaded, load it
        if db_name and db_name not in _palette_cache:
            _palette_cache[db_name] = _load_palette_cache(db_name)
            if norm_path in _palette_cache[db_name]:
                return _palette_cache[db_name][norm_path]
    
    return None


# ------------- DASH APP -------------
latent_options = get_latent_options()
db_options = get_db_options()
default_latent = latent_options[0]["value"] if latent_options else ""
default_db = db_options[0]["value"] if db_options else ""

app = dash.Dash(
    __name__,
    external_stylesheets=["https://codepen.io/chriddyp/pen/bWLwgP.css"],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1.0"}],
    suppress_callback_exceptions=True,
)


# attach route to the built-in Flask server
@app.server.route("/thumb")
def thumb_endpoint():
    p = request.args.get("p")
    if not p:
        return Response("missing p", status=400)
    # Validate against the servable allowlist before touching the filesystem.
    # 404 rather than 403 so this cannot be used to probe for files.
    # request.args is already percent-decoded by Flask; decoding again here
    # corrupted any filename containing a literal '%'.
    path = resolve_media_request(p)
    if path is None:
        return Response(status=404)
    data = make_thumbnail_bytes(path)
    if data is None:
        return Response(status=404)
    return Response(data, mimetype="image/jpeg", headers={"Cache-Control": "public, max-age=31536000"})


@app.server.route("/preview")
def preview_endpoint():
    p = request.args.get("p")
    if not p:
        return Response("missing p", status=400)
    # Validate against the servable allowlist before touching the filesystem.
    # 404 rather than 403 so this cannot be used to probe for files.
    # request.args is already percent-decoded by Flask; decoding again here
    # corrupted any filename containing a literal '%'.
    path = resolve_media_request(p)
    if path is None:
        return Response(status=404)

    # optional query params: width (w) and jpeg quality (q)
    try:
        max_w = min(max(200, int(request.args.get("w", 900))), 2048)
    except Exception:
        max_w = 900
    try:
        q = min(max(50, int(request.args.get("q", 72))), 95)
    except Exception:
        q = 72

    data = make_resized_bytes(path, max_w=max_w, q=q)
    if data is None:
        return Response(status=404)

    return Response(
        data,
        mimetype="image/jpeg",
        headers={"Cache-Control": "public, max-age=31536000"},
    )


@app.server.route("/palette")
def palette_endpoint():
    """Generate a color palette swatch image for an image."""
    p = request.args.get("p")
    if not p:
        return Response("missing p", status=400)
    # Validate against the servable allowlist before touching the filesystem.
    # 404 rather than 403 so this cannot be used to probe for files.
    # request.args is already percent-decoded by Flask; decoding again here
    # corrupted any filename containing a literal '%'.
    path = resolve_media_request(p)
    if path is None:
        return Response(status=404)
    
    # Number of colors
    try:
        n_colors = min(max(4, int(request.args.get("n", 16))), 32)
    except:
        n_colors = 16
    
    # Swatch dimensions
    try:
        width = min(max(100, int(request.args.get("w", 300))), 600)
        height = min(max(20, int(request.args.get("h", 30))), 60)
    except:
        width, height = 300, 30
    
    # Optional db_name for cache lookup
    db_name = request.args.get("db", None)
    
    try:
        # Try cached palette first (much faster)
        cached = get_cached_palette(path, db_name)
        if cached is not None:
            # Cached is (32, 4) with [L, A, B, proportion]
            # Take top n_colors by proportion
            palette = cached[:n_colors]
            colors = palette[:, :3]
            proportions = palette[:, 3]
            # Renormalize proportions
            proportions = proportions / (proportions.sum() + 1e-8)
        else:
            # Fallback: compute on the fly (slower)
            try:
                from .palette import extract_dominant_colors
            except ImportError:
                from palette import extract_dominant_colors
            
            palette = extract_dominant_colors(path, n_colors=n_colors)
            colors = palette[:, :3]
            proportions = palette[:, 3]
        
        # Colors are in LAB, convert to RGB for display
        # LAB values: L [0-100], A,B [-128 to 127]
        # Convert back to uint8 LAB then to RGB
        lab_colors = np.zeros((n_colors, 1, 3), dtype=np.float32)
        lab_colors[:, 0, :] = colors
        
        # Undo the float conversion: L back to 0-255, A,B back to 0-255
        lab_colors[:, :, 0] = lab_colors[:, :, 0] * (255.0 / 100.0)  # L: 0-100 -> 0-255
        lab_colors[:, :, 1] = lab_colors[:, :, 1] + 128.0             # A: -128,127 -> 0-255
        lab_colors[:, :, 2] = lab_colors[:, :, 2] + 128.0             # B: -128,127 -> 0-255
        lab_colors = np.clip(lab_colors, 0, 255).astype(np.uint8)
        
        # Convert LAB to BGR
        bgr_colors = cv2.cvtColor(lab_colors, cv2.COLOR_LAB2BGR)
        
        # Create swatch image
        swatch = np.zeros((height, width, 3), dtype=np.uint8)
        x = 0
        for i, prop in enumerate(proportions):
            w = int(prop * width)
            if i == len(proportions) - 1:
                w = width - x  # Fill remaining
            if w > 0:
                swatch[:, x:x+w] = bgr_colors[i, 0]
                x += w
        
        # Encode as PNG
        ok, buf = cv2.imencode(".png", swatch)
        if not ok:
            return Response(status=500)
        
        return Response(
            buf.tobytes(),
            mimetype="image/png",
            headers={"Cache-Control": "public, max-age=86400"},
        )
    except Exception as e:
        # Return a gray placeholder on error
        print(f"[palette endpoint] Error: {e}")
        swatch = np.full((height, width, 3), 64, dtype=np.uint8)
        ok, buf = cv2.imencode(".png", swatch)
        return Response(buf.tobytes(), mimetype="image/png") if ok else Response(status=500)


@app.server.route("/audio")
def audio_endpoint():
    p = request.args.get("p")
    if not p:
        return Response("missing p", status=400)
    # Validate against the servable allowlist before touching the filesystem.
    # 404 rather than 403 so this cannot be used to probe for files.
    # request.args is already percent-decoded by Flask; decoding again here
    # corrupted any filename containing a literal '%'.
    path = resolve_media_request(p)
    if path is None:
        return Response(status=404)
    full = resolve_path(path)
    if not os.path.exists(full):
        return Response(status=404)
    ext = os.path.splitext(full)[1].lower()
    mime = "audio/mpeg" if ext in [".mp3", ".m4a", ".aac"] else "audio/wav"
    with open(full, "rb") as f:
        data = f.read()
    return Response(data, mimetype=mime, headers={"Cache-Control": "public, max-age=31536000"})

@app.server.route("/awave")
def awave_endpoint():
    p = request.args.get("p")
    if not p:
        return Response("missing p", status=400)
    # Validate against the servable allowlist before touching the filesystem.
    # 404 rather than 403 so this cannot be used to probe for files.
    # request.args is already percent-decoded by Flask; decoding again here
    # corrupted any filename containing a literal '%'.
    path = resolve_media_request(p)
    if path is None:
        return Response(status=404)
    data = make_waveform_png(path)
    if data is None:
        return Response(status=404)
    return Response(data, mimetype="image/png", headers={"Cache-Control": "public, max-age=31536000"})

@app.server.route("/aspec")
def aspec_endpoint():
    p = request.args.get("p")
    if not p:
        return Response("missing p", status=400)
    # Validate against the servable allowlist before touching the filesystem.
    # 404 rather than 403 so this cannot be used to probe for files.
    # request.args is already percent-decoded by Flask; decoding again here
    # corrupted any filename containing a literal '%'.
    path = resolve_media_request(p)
    if path is None:
        return Response(status=404)
    data = make_melspec_png(path)
    if data is None:
        return Response(status=404)
    return Response(data, mimetype="image/png", headers={"Cache-Control": "public, max-age=31536000"})


@app.callback(
    Output({"type": "select-image", "index": ALL}, "on"),
    Input("select-all", "n_clicks"),
    Input("clear-all", "n_clicks"),
    State({"type": "select-image", "index": ALL}, "on"),
    prevent_initial_call=True,
)
def bulk_select(n_all, n_clear, current_states):
    # No images rendered yet
    if not isinstance(current_states, list):
        return dash.no_update

    trig = ctx.triggered_id
    if trig == "select-all":
        return [True] * len(current_states)
    if trig == "clear-all":
        return [False] * len(current_states)
    return dash.no_update


# The hover preview's geometry lives here so the layout and the callback cannot
# disagree. The callback used to return its own position:fixed/top:100px/left:100px,
# which pinned the preview over the mode menu regardless of what the layout said.
HOVER_THUMB_STYLE = {
    "position": "absolute",
    "top": "6px",
    # Left, not right: the plot's top-right corner already holds both the
    # Plotly modebar and the cluster legend, and the preview covered the
    # legend. The top-left is the only uncontested corner -- and because this
    # is absolute inside #scatter-wrapper rather than fixed to the window, it
    # still cannot reach the mode menu above the plot.
    "left": "6px",
    "zIndex": 20,
    "maxWidth": "170px",
    "maxHeight": "130px",
    "borderRadius": "6px",
    "border": "1px solid #ffffff55",
    "boxShadow": "0 6px 20px #000c",
    "backgroundColor": "#000",
    "objectFit": "contain",
    "pointerEvents": "none",
}
HOVER_THUMB_HIDDEN = {**HOVER_THUMB_STYLE, "display": "none"}
HOVER_THUMB_SHOWN = {**HOVER_THUMB_STYLE, "display": "block"}



def _blank_fig(message="Choose a dataset above to begin."):
    """
    The figure shown before a dataset is picked, and whenever one cannot be
    loaded.

    A bare go.Figure() renders with Plotly's default white paper, which on this
    dark page is a large white rectangle -- the first thing you see on launch.
    This matches the loaded plot's colours and says what to do next instead of
    showing empty axes with meaningless 0..6 ticks.
    """
    fig = go.Figure()
    fig.update_layout(
        plot_bgcolor="#121212",
        paper_bgcolor="#121212",
        font=dict(color="#9a9aa2"),
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        annotations=[dict(
            text=message, showarrow=False, xref="paper", yref="paper",
            x=0.5, y=0.5, font=dict(size=13, color="#6c6c74"),
        )],
    )
    return fig


app.layout = html.Div(
    # A fixed-height flex column, so the app fills the window exactly instead
    # of overflowing it by the height of the header. The plot flexes; the
    # controls under it are pinned; only the side panel scrolls, and only
    # when its own content needs it.
    style={"backgroundColor": "#121212", "color": "white",
           "padding": "14px 18px 10px", "height": "100vh",
           "boxSizing": "border-box", "overflow": "hidden",
           "display": "flex", "flexDirection": "column", "gap": "8px"},
    children=[
        html.Div(
            [
                # The mark, then the tabs. Dash serves anything in
                # arcana/assets, so this survives freezing as long as the spec
                # ships that directory -- which it already does for custom.css
                # and the label lists.
                html.Div(
                    style={"display": "flex", "alignItems": "center", "gap": "11px",
                           "flex": "0 0 auto", "marginRight": "28px"},
                    children=[
                        html.Img(
                            src=app.get_asset_url("arcana-mark.png"),
                            alt="",
                            style={"height": "36px", "width": "auto",
                                   "display": "block"},
                        ),
                        # The name is set in the app's own type rather than
                        # taken from the logo file: the wordmark there is a pale
                        # outline drawn for a light page, and on this ground it
                        # reads as haze.
                        html.Span("Arcana", style={
                            "fontSize": "19px", "fontWeight": "600",
                            "letterSpacing": "0.5px", "color": _ui.INK}),
                    ],
                ),
                dcc.RadioItems(
                    id="mode-select",
                    # Datasets first, and selected by default. It is the only
                    # tab that does anything useful with no dataset indexed --
                    # the other three open on an empty plot and an empty
                    # dropdown, which is a poor first thing to see and gives no
                    # hint that indexing is the missing step.
                    options=[
                        {"label": "Datasets", "value": "datasets"},
                        {"label": "Prompt Search", "value": "prompt"},
                        {"label": "Generate Story", "value": "story"},
                        {"label": "Moodboard", "value": "moodboard"},
                    ],
                    value="datasets",
                    # Dash 4 gives labels its own dark design-token colour, which is
                    # invisible on this dark ground. Set it explicitly so the app does
                    # not depend on any external or default stylesheet.
                    labelStyle={"display": "inline-block", "marginRight": "25px",
                                "fontWeight": "bold", "color": "#fff", "cursor": "pointer"},
                    inputStyle={"marginRight": "6px", "accentColor": "#00bcd4"},
                ),
                html.Label("Dataset:", style={"marginLeft": "40px", "marginRight": "6px"}),
                dcc.Dropdown(
                    id="dataset-dropdown",
                    options=get_matching_datasets(),
                    value=None,  # set default below after checking
                    clearable=False,
                    style={"width": "220px", "display": "inline-block", "verticalAlign": "middle", "color": "#000"},
                ),
            ],
            style={"display": "flex", "alignItems": "center", "marginBottom": "20px"},
        ),

        # ── Relocation panel ────────────────────────────────────────────────
        # Appears only when the selected dataset's files are not where its index
        # says they are -- an external drive that is not plugged in, a folder
        # that moved, a library copied to another machine. Nothing is re-encoded:
        # items are matched by content, so this is seconds, not hours.
        html.Div(
            [
                html.Div(id="relocate-message",
                         style={"fontSize": "13px", "color": "#e0a44a", "marginBottom": "8px"}),
                html.Div(
                    [
                        dcc.Input(
                            id="relocate-root",
                            type="text",
                            placeholder="Paste the folder where these files live now...",
                            debounce=True,
                            style={"flex": "1", "marginRight": "8px", "minWidth": "260px"},
                        ),
                        html.Button("Check", id="relocate-check-btn", n_clicks=0,
                                    style=_ui.button("secondary", marginRight="6px")),
                        html.Button("Relocate", id="relocate-apply-btn", n_clicks=0,
                                    style=_ui.button("success")),
                    ],
                    style={"display": "flex", "alignItems": "center"},
                ),
                html.Div(id="relocate-status",
                         style={"fontSize": "12px", "marginTop": "8px", "color": "#888"}),
            ],
            id="relocate-panel",
            style={"display": "none"},
        ),

        _ui_datasets.layout(),

        dcc.Store(id="story-cache", storage_type="memory"),
        dcc.Store(id="grouped-results", storage_type="memory"),
        dcc.Store(id="carousel-state", storage_type="memory"),
        dcc.Store(id="carousel-order", storage_type="memory"),
        dcc.Store(id="results-owner", storage_type="memory"),  # which mode filled image-display
        dcc.Store(id="moodboard-store", storage_type="local"),  # Persist moodboard across sessions
        dcc.Store(id="selected-moodboard-image", storage_type="memory"),  # Reference image (palette source for search/transfer)
        dcc.Store(id="selected-target-image", storage_type="memory"),  # Target image (receives colors in transfer)

        # Everything below the header shares the remaining height. min-height:0
        # is what actually lets a flex child shrink instead of forcing the page
        # taller than the window.
        html.Div(id="main-row", style={"flex": "1", "minHeight": "0",
                                       "display": "flex", "gap": "18px"},
                 children=[
        # ── THE RAIL ────────────────────────────────────────────────────────
        # The kept-image collection, evacuated from the right column. It used to
        # sit inside #moodboard-section above the search results, which pushed
        # #image-display to y~763 of an 812px column -- pressing "Find Similar"
        # at top-left produced about 49 visible pixels of result in the far
        # corner. Out here it is a fixed 240px band immediately left of the tool
        # card, so whichever tool is live, the pictures it draws from are one
        # 18px gutter away. Ground is BG, the most recessed surface: this is
        # storage, not action.
        html.Div(
            id="moodboard-rail",
            style={"display": "none"},
            children=[
                html.Div(id="moodboard-rail-count",
                         children="COLLECTION",
                         style={"fontSize": "11px", "fontWeight": "700",
                                "letterSpacing": "1px", "color": _ui.INK_DIM,
                                "flex": "0 0 auto", "marginBottom": "10px"}),
                dcc.Upload(
                    id="moodboard-external-upload",
                    children=html.Div("Drop images to collect them here",
                                      style={"textAlign": "center", "lineHeight": "1.35"}),
                    style={"padding": "10px", "border": f"2px dashed {_ui.LINE}",
                           "borderRadius": "8px", "backgroundColor": _ui.SURFACE,
                           "cursor": "pointer", "fontSize": "11.5px",
                           "color": _ui.INK_DIM, "flex": "0 0 auto"},
                    style_active={"borderColor": _ui.INK_DIM},
                    accept="image/*",
                    multiple=False,
                ),
                # The only flexible item in the rail, so every rounding error in
                # the column lands here rather than clipping the footer.
                html.Div(id="moodboard-gallery", style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(auto-fill, minmax(78px, 1fr))",
                    "gap": "8px",
                    "flex": "1 1 0",
                    "minHeight": "0",
                    "overflowY": "auto",
                    "overflowX": "hidden",
                    "padding": "10px 2px",
                    # Rows size to their content. Without this the grid stretches
                    # each row to fill the column, and every thumbnail's
                    # absolutely-positioned x badge lands at the bottom of the
                    # rail instead of on its own picture.
                    "alignContent": "start",
                    "alignItems": "start",
                }),
                # A board is the collection plus which picture is [R] and which
                # is [T], saved by name in the data directory. The collection
                # used to live only in browser localStorage: one unnamed list,
                # no way back to yesterday's, and Clear emptied it with no undo.
                html.Div(
                    [
                        html.Div("BOARD", style={"fontSize": "10px", "color": "#888",
                                                 "letterSpacing": "0.5px",
                                                 "marginBottom": "4px"}),
                        dcc.Dropdown(
                            id="board-picker", placeholder="Saved boards...",
                            options=[], value=None, clearable=True,
                            style={"width": "100%", "minWidth": "0",
                                   "fontSize": "12px"},
                        ),
                        html.Div(
                            [
                                html.Button("Load", id="board-load-btn", n_clicks=0,
                                            style=_ui.button("secondary",
                                                             padding="4px 10px",
                                                             fontSize="12px")),
                                html.Button("Delete", id="board-delete-btn", n_clicks=0,
                                            style=_ui.button("secondary",
                                                             padding="4px 10px",
                                                             fontSize="12px")),
                            ],
                            style={"display": "flex", "gap": "6px",
                                   "marginTop": "6px"},
                        ),
                        dcc.Input(
                            id="board-name", type="text",
                            placeholder="Name this board...",
                            style=_ui.input_box(width="100%", marginTop="8px",
                                                padding="6px 9px", fontSize="12px",
                                                boxSizing="border-box"),
                        ),
                        html.Button("Save board", id="board-save-btn", n_clicks=0,
                                    style=_ui.button("success", padding="5px 11px",
                                                     fontSize="12px",
                                                     marginTop="6px", width="100%")),
                        html.Div(id="board-status",
                                 style={"fontSize": "11.5px", "color": _ui.INK_DIM,
                                        "minHeight": "16px", "marginTop": "6px",
                                        "lineHeight": "1.4"}),
                    ],
                    style={"flex": "0 0 auto", "marginTop": "10px",
                           "paddingTop": "10px",
                           "borderTop": "1px solid #333"},
                ),
                html.Div(
                    [
                        html.Button("Clear", id="clear-moodboard-btn", n_clicks=0,
                                    style=_ui.button("secondary", padding="5px 11px",
                                                     fontSize="12px")),
                        # Renamed: this copies the image FILES out to a folder,
                        # which is a different thing from saving a board, and
                        # calling both "save" is how it read before.
                        html.Button("Export copies", id="save-moodboard-btn", n_clicks=0,
                                    style=_ui.button("secondary", padding="5px 11px",
                                                     fontSize="12px")),
                    ],
                    style={"display": "flex", "gap": "8px", "flex": "0 0 auto",
                           "marginTop": "10px"},
                ),
                dcc.Input(
                    id="moodboard-save-folder",
                    type="text",
                    placeholder="Export folder name...",
                    style=_ui.input_box(width="100%", marginTop="8px",
                                        padding="6px 9px", fontSize="12px",
                                        boxSizing="border-box"),
                ),
                # Height reserved so an async confirmation cannot reflow the rail.
                html.Div(id="moodboard-save-confirmation",
                         style={"fontSize": "11.5px", "color": _ui.OK,
                                "minHeight": "16px", "marginTop": "6px",
                                "flex": "0 0 auto"}),
            ],
        ),
        html.Div(
            [
                html.Div(
                    id="scatter-wrapper",
                    # position:relative anchors the hover preview below to the
                    # plot rather than to the window, where it used to sit on
                    # top of the mode menu.
                    style={"flex": "1", "minHeight": "0", "position": "relative"},
                    children=[
                        dcc.Graph(id="scatter-plot", figure=_blank_fig(),
                                  style={"height": "100%"},
                                  config={"responsive": True}),
                        html.Img(id="hover-thumb", style=HOVER_THUMB_HIDDEN),
                    ],
                ),
                html.Div(
                    [
                        dcc.Input(
                            id="search-box",
                            type="text",
                            placeholder="Enter a prompt...",
                            style={"width": "60%", "marginRight": "10px"},
                        ),
                        dcc.Input(
                            id="num-images",
                            type="number",
                            value=4,
                            min=1,
                            max=1000,
                            style={"width": "15%", "marginRight": "10px"},
                        ),
                        dcc.Textarea(
                            id="story-box",
                            placeholder="Enter your story, one scene per line. (Press ENTER after each scene.)",
                            style={"width": "70%", "height": "70px", "marginRight": "10px"},
                        ),
                        html.Button("Search", id="main-action-btn", n_clicks=0, style=_ui.button("primary")),
                    ],
                    id="controls-bar",
                    style={"display": "flex", "alignItems": "center",
                           "marginTop": "10px", "flex": "0 0 auto"},
                ),
                # Moodboard controls (hidden by default)
                html.Div(
                    [
                        # THE TOOL SWITCHER. Both cards mounted together are
                        # ~930px against the ~810px the bench has, which is what
                        # forced the bench to scroll and put "Find Similar" and
                        # the Color Transfer panel at opposite ends of it. They
                        # share no settings and write to the same output region,
                        # so showing one at a time costs nothing and makes the
                        # stage unambiguous about which control produced what.
                        dcc.RadioItems(
                            id="moodboard-tool",
                            options=[
                                {"label": "Find similar", "value": "search"},
                                {"label": "Colour transfer", "value": "transfer"},
                            ],
                            value="search",
                            className="tool-switch",
                            labelStyle={"cursor": "pointer"},
                            style={"marginBottom": "12px"},
                        ),
                        # ─────────────────────────────────────────────────────────
                        # SIMILARITY SEARCH PANEL
                        # ─────────────────────────────────────────────────────────
                        html.Div(
                            [
                                # Header Row with Upload Zone
                                html.Div(
                                    [
                                        html.Div("🔍 Similarity Search", style={"fontSize": "14px", "fontWeight": "600", "color": "#00bcd4"}),
                                        # The drop zone that used to live here is gone.
                                        # It wrote to the same store as the gallery's,
                                        # so one drop became the search query AND the
                                        # colour-transfer palette source. There is now a
                                        # single intake for the page, in the collection
                                        # rail, and roles are assigned from the pictures.
                                        html.Span("uses the [R] picture in the collection",
                                                  style={"fontSize": "11px", "color": "#7a8a9a"}),
                                    ],
                                    style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "12px"},
                                ),
                        
                        # Feature Cards Row
                        html.Div(
                            [
                                # Color Palette Card
                                html.Div(
                                    [
                                        html.Div([
                                            dcc.Checklist(
                                                id="moodboard-palette-check",
                                                options=[{"label": "🎨 Color Palette", "value": "palette"}],
                                                value=["palette"],
                                                inline=True,
                                                style={"fontWeight": "500"},
                                            ),
                                            html.Span(id="palette-db-status", style={"fontSize": "10px", "marginLeft": "8px"}),
                                        ], style={"marginBottom": "8px", "display": "flex", "alignItems": "center"}),
                                        html.Div([
                                            html.Span("Method", style={"fontSize": "10px", "color": "#888", "marginBottom": "4px", "textTransform": "uppercase", "letterSpacing": "0.5px"}),
                                            dcc.Dropdown(
                                                id="palette-method",
                                                options=[
                                                    {"label": "EMD", "value": "emd"},
                                                    {"label": "Histogram", "value": "histogram"},
                                                    {"label": "Moments", "value": "moments"},
                                                ],
                                                # EMD matches palettes by how far
                                                # colours actually sit from each other in
                                                # LAB; the histogram compares fixed bins,
                                                # so two near-identical colours either
                                                # side of a bin edge score as unrelated.
                                                # It costs 2.7s against 0.14s across 9,359
                                                # images, which is worth it at the sizes
                                                # this app is used at.
                                                #
                                                # It also reads `dominant`, the one palette
                                                # feature the LAB double-conversion never
                                                # touched, so it gives full-quality results
                                                # on datasets indexed before that fix.
                                                value="emd",
                                                clearable=False,
                                                style={"width": "100%", "minWidth": "0", "fontSize": "12px"},
                                            ),
                                        ], style={"display": "flex", "flexDirection": "column"}),
                                        html.Div([
                                            html.Span("Colors", style={"fontSize": "10px", "color": "#888", "marginBottom": "4px", "textTransform": "uppercase", "letterSpacing": "0.5px"}),
                                            dcc.Dropdown(
                                                id="palette-n-colors",
                                                options=[{"label": str(n), "value": n} for n in [4, 8, 12, 16, 24, 32]],
                                                value=16,
                                                clearable=False,
                                                style={"width": "100%", "minWidth": "0", "fontSize": "12px"},
                                            ),
                                        ], style={"display": "flex", "flexDirection": "column", "marginTop": "8px"}),
                                    ],
                                    id="palette-card",
                                    style={"padding": "10px", "backgroundColor": "#252525", "borderRadius": "8px", "border": "1px solid #333", "flex": "1 1 0", "minWidth": "128px", "overflow": "hidden"},
                                ),
                                # Style Card
                                html.Div(
                                    [
                                        html.Div([
                                            dcc.Checklist(
                                                id="moodboard-style-check",
                                                options=[{"label": "✨ Style/Texture", "value": "style"}],
                                                value=[],
                                                inline=True,
                                                style={"fontWeight": "500"},
                                            ),
                                            html.Span(id="style-db-status", style={"fontSize": "10px", "marginLeft": "8px"}),
                                        ], style={"marginBottom": "8px", "display": "flex", "alignItems": "center"}),
                                        html.Div([
                                            html.Span("Method", style={"fontSize": "10px", "color": "#888", "marginBottom": "4px", "textTransform": "uppercase", "letterSpacing": "0.5px"}),
                                            dcc.Dropdown(
                                                id="style-method",
                                                options=[
                                                    {"label": "Gram", "value": "gram"},
                                                    {"label": "Edge", "value": "edge"},
                                                    {"label": "LBP", "value": "lbp"},
                                                ],
                                                value="gram",
                                                clearable=False,
                                                style={"width": "100%", "minWidth": "0", "fontSize": "12px"},
                                            ),
                                        ], style={"display": "flex", "flexDirection": "column"}),
                                    ],
                                    id="style-card",
                                    style={"padding": "10px", "backgroundColor": "#252525", "borderRadius": "8px", "border": "1px solid #333", "flex": "1 1 0", "minWidth": "128px", "overflow": "hidden"},
                                ),
                            ],
                            style={"display": "flex", "gap": "10px", "marginBottom": "12px", "flexWrap": "wrap"},
                        ),
                        
                        # Search Options Row
                        html.Div(
                            [
                                dcc.Input(
                                    id="moodboard-prompt",
                                    type="text",
                                    placeholder="Filter by prompt (optional)...",
                                    style=_ui.input_box(flex="1", minWidth="150px"),
                                ),
                                html.Div([
                                    html.Span("Results:", style={"fontSize": "11px", "color": "#888", "marginRight": "4px"}),
                                    dcc.Input(
                                        id="moodboard-num",
                                        type="number",
                                        value=50,
                                        min=1,
                                        max=500,
                                        style=_ui.input_box(width="60px", padding="6px", textAlign="center"),
                                    ),
                                ], style={"display": "flex", "alignItems": "center"}),
                            ],
                            style={"display": "flex", "gap": "10px", "alignItems": "center", "marginBottom": "12px", "flexWrap": "wrap"},
                        ),
                        
                        # Display Options Row
                        html.Div(
                            [
                                dcc.Checklist(
                                    id="show-palette-swatches",
                                    options=[{"label": "Show palettes", "value": "show"}],
                                    value=["show"],
                                    inline=True,
                                    style={"fontSize": "12px"},
                                ),
                                html.Div([
                                    html.Span("Image size:", style={"fontSize": "11px", "color": "#888", "marginRight": "6px"}),
                                    dcc.Dropdown(
                                        id="moodboard-img-size",
                                        options=[
                                            {"label": "Small", "value": "small"},
                                            {"label": "Medium", "value": "medium"},
                                            {"label": "Large", "value": "large"},
                                            {"label": "Full", "value": "full"},
                                        ],
                                        value="medium",
                                        clearable=False,
                                        style={"width": "90px", "color": "#000", "fontSize": "12px"},
                                    ),
                                ], style={"display": "flex", "alignItems": "center"}),
                                html.Div([
                                    html.Span("Columns:", style={"fontSize": "11px", "color": "#888", "marginRight": "6px"}),
                                    dcc.Dropdown(
                                        id="moodboard-columns",
                                        options=[{"label": str(n), "value": n} for n in [1, 2, 3, 4]],
                                        value=2,
                                        clearable=False,
                                        style={"width": "55px", "color": "#000", "fontSize": "12px"},
                                    ),
                                ], style={"display": "flex", "alignItems": "center"}),
                                html.Button(
                                    "Find Similar", 
                                    id="moodboard-search-btn", 
                                    n_clicks=0,
                                    style=_ui.button("primary"),
                                ),
                            ],
                            style={"display": "flex", "gap": "12px", "alignItems": "center", "flexWrap": "wrap"},
                        ),
                            ],
                            id="similarity-card",
                            style={"padding": "14px", "backgroundColor": "#0a1a1a", "borderRadius": "8px", "border": "1px solid #2a4a4a"},
                        ),
                        
                        # ─────────────────────────────────────────────────────────
                        # COLOR TRANSFER PANEL
                        # ─────────────────────────────────────────────────────────
                        html.Div(
                            [
                                html.Div("🎨 Color Transfer", style={"fontSize": "14px", "fontWeight": "600", "color": "#e040fb", "marginBottom": "10px"}),
                                html.Div("Apply Reference palette to Target image", style={"fontSize": "11px", "color": "#888", "marginBottom": "12px"}),
                                html.Div(
                                    [
                                        # Method dropdown
                                        html.Div([
                                            html.Span("Method:", style={"fontSize": "11px", "color": "#888", "marginRight": "6px"}),
                                            dcc.Dropdown(
                                                id="color-transfer-method",
                                                options=[
                                                    {"label": "ModFlows (Neural)", "value": "modflows"},
                                                    {"label": "LAB (Reinhard)", "value": "lab"},
                                                ],
                                                value="modflows",
                                                clearable=False,
                                                style={"width": "140px", "color": "#000", "fontSize": "12px"},
                                            ),
                                        ], style={"display": "flex", "alignItems": "center"}),
                                        # Strength slider
                                        html.Div([
                                            html.Span("Strength:", style={"fontSize": "11px", "color": "#888", "marginRight": "8px"}),
                                            dcc.Slider(
                                                id="color-transfer-strength",
                                                min=0.0, max=1.0, step=0.1, value=1.0,
                                                marks={0: "0", 0.5: "0.5", 1: "1"},
                                                tooltip={"placement": "bottom", "always_visible": False},
                                            ),
                                        ], style={"flex": "1", "minWidth": "150px"}),
                                        # ── QUALITY ─────────────────────────
                                        # One choice replaces Size, Steps and a
                                        # "Full-res output" checkbox. Those were
                                        # three implementation details the user
                                        # had to understand, and two of them
                                        # decided whether the result came back
                                        # at 1024px or at the original size.
                                        # Every preset now returns the picture
                                        # at its original resolution: the flow
                                        # runs small, because it is looking for
                                        # a global colour mapping and does not
                                        # need detail to find one, and that
                                        # mapping is applied at full size
                                        # through a 3-D LUT.
                                        html.Div([
                                            html.Span("Quality:", style={"fontSize": "11px", "color": "#888", "marginRight": "6px"}),
                                            dcc.Dropdown(
                                                id="color-transfer-quality",
                                                options=[
                                                    {"label": "Quick look", "value": "quick"},
                                                    {"label": "Balanced", "value": "balanced"},
                                                    {"label": "Best", "value": "best"},
                                                ],
                                                value="balanced",
                                                clearable=False,
                                                style={"width": "150px", "color": "#000", "fontSize": "12px"},
                                            ),
                                        ], id="color-transfer-quality-wrapper",
                                           style={"display": "flex", "alignItems": "center"}),
                                        html.Div(id="color-transfer-quality-note",
                                                 style={"fontSize": "11px", "color": _ui.INK_DIM,
                                                        "width": "100%", "marginTop": "-4px",
                                                        "minHeight": "15px"}),
                                        # ── the ModFlows model, if it is missing ──
                                        # The neural method needs a 229 MB
                                        # checkpoint that is deliberately not
                                        # shipped in the build. Without this the
                                        # only sign of that was an error message
                                        # after pressing Transfer.
                                        html.Div(id="ct-model-status",
                                                 style={"marginBottom": "10px"}),
                                        dcc.Interval(id="ct-poll", interval=700,
                                                     disabled=True),
                                        dcc.Store(id="ct-job",
                                                  storage_type="memory"),
                                        # ── WHICH COLOURS OF THE REFERENCE ──
                                        # Both methods read the reference's whole
                                        # colour distribution, so a picture that
                                        # reads as teal but is 70% near-black
                                        # transfers as mostly-black. These two
                                        # ranges pick the pixels that actually get
                                        # used; the swatches below show the result
                                        # before you spend 30s on a transfer.
                                        html.Details([
                                            html.Summary(
                                                "Which colours of the reference",
                                                style={"fontSize": "11.5px",
                                                       "color": "#00bcd4",
                                                       "cursor": "pointer",
                                                       "marginBottom": "8px",
                                                       "userSelect": "none"},
                                            ),
                                            html.Div("Lightness", style=_ui.field_label()),
                                            dcc.RangeSlider(
                                                id="ct-lightness",
                                                min=0, max=100, step=1, value=[0, 100],
                                                marks={0: {"label": "black"},
                                                       50: {"label": "mid"},
                                                       100: {"label": "white"}},
                                                tooltip={"placement": "bottom"},
                                            ),
                                            html.Div("Saturation", style=_ui.field_label(
                                                marginTop="10px")),
                                            dcc.RangeSlider(
                                                id="ct-saturation",
                                                min=0, max=100, step=1, value=[0, 100],
                                                marks={0: {"label": "grey"},
                                                       100: {"label": "vivid"}},
                                                tooltip={"placement": "bottom"},
                                            ),
                                            html.Div(id="ct-swatches",
                                                     style={"display": "flex", "height": "22px",
                                                            "borderRadius": "4px",
                                                            "overflow": "hidden",
                                                            "marginTop": "12px",
                                                            "border": f"1px solid {_ui.LINE}"}),
                                            html.Div(id="ct-keep-note",
                                                     style={"fontSize": "11px",
                                                            "color": _ui.INK_DIM,
                                                            "marginTop": "6px",
                                                            "minHeight": "15px"}),
                                            html.Button(
                                                "Reset", id="ct-reset", n_clicks=0,
                                                style=_ui.button("ghost", padding="4px 0",
                                                                 fontSize="11px",
                                                                 marginTop="4px"),
                                            ),
                                        ], style={"marginTop": "6px", "marginBottom": "10px",
                                                  "borderTop": f"1px solid {_ui.LINE}",
                                                  "paddingTop": "10px"}),
                                        html.Button(
                                            "Transfer Colors",
                                            id="color-transfer-btn",
                                            n_clicks=0,
                                            style=_ui.button("primary"),
                                        ),
                                    ],
                                    style={"display": "flex", "gap": "16px", "alignItems": "center", "flexWrap": "wrap"},
                                ),
                                html.Div(id="color-transfer-status", style={"marginTop": "10px", "fontSize": "12px"}),
                            ],
                            id="transfer-card",
                            style={"marginTop": "16px", "padding": "14px", "backgroundColor": "#1a1a2a", "borderRadius": "8px", "border": "1px solid #4a3a5a"},
                        ),
                    ],
                    id="moodboard-controls",
                    style={"display": "none", "marginTop": "10px", "padding": "16px", "backgroundColor": "#1a1a1a", "borderRadius": "10px", "border": "1px solid #333"},
                ),
                # Reference image display (sticky in left column)
                html.Div(
                    id="moodboard-ref-display",
                    style={"display": "none", "marginTop": "12px", "padding": "12px", "backgroundColor": "#1a1a1a", "borderRadius": "10px", "border": "1px solid #00bcd4", "position": "sticky", "top": "10px"},
                ),
            ],
            id="left-column",
            style={"flex": "1 1 58%", "minWidth": "0", "minHeight": "0",
                   "display": "flex", "flexDirection": "column"},
        ),

        # ───────────────────────── RIGHT COLUMN ─────────────────────────
html.Div(
    [
        # Controls panel “card”
        html.Div(
            [
                # ONE ROW: Poetry (when Story+Image) + Grouping + (Audio) Spectrogram
                html.Div(
                    [
                        # Poetry inline (story+image only)
                        html.Div(
                            [
                                html.Button(
                                    "Inject Poetry",
                                    id="inject-poetry-btn",
                                ),
                                daq.Knob(
                                    id="poetry-strength",
                                    value=0.72,
                                    min=0.0,
                                    max=1.0,
                                    size=60,
                                    color="#00bcd4",
                                    label="Strength",
                                ),
                            ],
                            id="poetry-inline",
                            style={
                                "display": "none",           # hidden until Story+Image
                                "alignItems": "center",
                                "gap": "10px",
                            },
                        ),

                        # Inject Poetry runs Stable Diffusion once per scene.
                        # Measured at 1024px and 4 steps: 1.3 s an image on the
                        # GPU, 37.5 s on the CPU -- and the packaged build is
                        # CPU-only, so a five-scene story is about three minutes
                        # there. It used to report nothing at all for that whole
                        # time, which is indistinguishable from a dead button.
                        html.Div(
                            id="poetry-progress",
                            style={"display": "none", "flexDirection": "column",
                                   "gap": "6px", "minWidth": "220px"},
                            children=[
                                html.Div(id="poetry-progress-msg",
                                         style={"fontSize": "11px", "color": "#bdbdbd"}),
                                html.Div(
                                    style={"height": "6px", "borderRadius": "3px",
                                           "backgroundColor": "#333", "overflow": "hidden"},
                                    children=html.Div(
                                        id="poetry-progress-bar",
                                        style={"height": "100%", "width": "0%",
                                               "backgroundColor": "#00bcd4",
                                               "transition": "width 0.3s"}),
                                ),
                            ],
                        ),
                        dcc.Interval(id="poetry-poll", interval=700, disabled=True),
                        dcc.Store(id="poetry-job"),

                        # Grouping controls: vertical layout
                        html.Div(
                            [
                                html.Div("Group twins",
                                        style={"fontSize": "12px", "textAlign": "center", "marginBottom": "4px"}),
                                daq.BooleanSwitch(id="group-similar", on=True, color="#00bcd4"),
                            ],
                            style={
                                "display": "flex",
                                "flexDirection": "column",
                                "alignItems": "center",
                                "minWidth": "80px",
                                "marginLeft": "6px",
                                "marginRight": "6px",
                            },
                        ),

                        html.Div(
                            [
                                html.Div("distance ≤",
                                        style={"fontSize": "12px", "textAlign": "center", "marginBottom": "4px"}),
                                dcc.Input(
                                    id="sim-thresh",
                                    type="number",
                                    value=0.08,
                                    step=0.01,
                                    min=0.0,
                                    max=0.5,
                                    style={"width": "76px", "color": "#000", "textAlign": "center"},
                                ),
                            ],
                            style={
                                "display": "flex",
                                "flexDirection": "column",
                                "alignItems": "center",
                                "minWidth": "80px",
                                "marginLeft": "6px",
                                "marginRight": "6px",
                            },
                        ),



                        # Spectrogram inline (audio only)
                        html.Div(
                            [
                                html.Span("Spectrogram", style={"marginRight": "8px", "whiteSpace": "nowrap"}),
                                daq.BooleanSwitch(id="spec-toggle", on=False, color="#00bcd4"),
                            ],
                            id="audio-spec-inline",
                            style={
                                "display": "none",           # shown for audio datasets
                                "alignItems": "center",
                                "gap": "10px",
                                "marginLeft": "12px",
                            },
                        ),
                    ],
                    style={
                        "display": "flex",
                        "alignItems": "center",
                        "gap": "12px",
                        "flexWrap": "wrap",
                        "margin": "0 0 10px 0",
                    },
                ),

                # THE RESULTS BAR. What used to live here was a single
                # "MOODBOARD IMAGES" card that held the kept-image collection AND,
                # below an <hr>, the controls that act on the *search results* --
                # two unrelated collections in one container, with the selection
                # buttons ~400px above the results they act on. The collection has
                # moved out to #moodboard-rail; what remains is only the bar that
                # governs the results, and it sticks to the top of the results
                # scrollport so it can never be scrolled away from them.
                html.Div(
                    [
                        html.Div(
                            [
                                html.Span("RESULTS", id="moodboard-results-count",
                                          style={"fontWeight": "700", "fontSize": "12px",
                                                 "letterSpacing": "1px", "color": _ui.INK}),
                                html.Button("Select all", id="moodboard-select-all", n_clicks=0,
                                            style=_ui.button("secondary", padding="5px 11px",
                                                             fontSize="12px")),
                                html.Button("Clear", id="moodboard-clear-all", n_clicks=0,
                                            style=_ui.button("secondary", padding="5px 11px",
                                                             fontSize="12px")),
                                dcc.Input(
                                    id="moodboard-results-folder",
                                    type="text",
                                    placeholder="Folder to save selected results...",
                                    style=_ui.input_box(flex="1", minWidth="140px",
                                                        padding="5px 9px", fontSize="12px"),
                                ),
                                html.Button("Save selected", id="moodboard-save-selected", n_clicks=0,
                                            style=_ui.button("success", padding="5px 11px",
                                                             fontSize="12px")),
                            ],
                            style={"display": "flex", "alignItems": "center", "gap": "8px",
                                   "flexWrap": "wrap"},
                        ),
                        html.Div(id="moodboard-results-confirmation",
                                 style={"fontSize": "12px", "color": _ui.OK}),
                    ],
                    id="moodboard-section",
                    # Sticky against #right-column's scrollport. The negative side
                    # margins bleed the bar through the wrapper's padding so it
                    # spans the full stage width instead of leaving gutters.
                    style={"display": "none", "position": "sticky", "top": "0",
                           "zIndex": 5, "margin": "0 -12px 12px", "padding": "9px 12px",
                           "backgroundColor": _ui.BG,
                           "borderBottom": f"2px solid {_ui.ACCENT}"},
                ),

                # Results list
                html.Div(
                    id="image-display",
                    style={"overflowX": "hidden"},
                ),

                # The colour-transfer output. Same region as the search results
                # -- only one of the two is ever mounted, so the stage always
                # has exactly one plausible author.
                html.Div(id="transfer-preview", style={"display": "none"}),

                # Bulk selection buttons (hidden in moodboard mode)
                html.Div(
                    [
                        html.Button("Select All", id="select-all", n_clicks=0,
                                    style=_ui.button("secondary", padding="6px 12px", fontSize="12px")),
                        html.Button("Clear All", id="clear-all", n_clicks=0,
                                    style=_ui.button("secondary", padding="6px 12px", fontSize="12px")),
                    ],
                    id="bulk-selection-btns",
                    style={"display": "flex", "gap": "8px", "marginTop": "8px"},
                ),

                # Save actions (hidden in moodboard mode)
                html.Div(
                    [
                        html.Button("Save Selected Images", id="save-button",
                                    style=_ui.button("success", marginTop="8px")),
                        dcc.Input(
                            id="save-folder",
                            type="text",
                            placeholder="Enter folder path...",
                            style={"width": "100%", "marginTop": "6px"},
                        ),
                    ],
                    id="save-actions-section",
                ),
                html.Button("Save Story", id="save-story-btn",
                            style=_ui.button("success", marginTop="10px", display="none")),
                html.Div(id="save-confirmation", style={"marginTop": "10px"}),
                html.Div(id="moodboard-added-notification", style={"marginTop": "6px"}),
            ],
            style={
                "backgroundColor": "#1b1b1b",
                "border": "1px solid #2a2a2a",
                "borderRadius": "10px",
                "padding": "12px",
                "marginTop": "6px",
            },
        ),
    ],
    id="right-column",
    style={"flex": "0 0 40%", "minWidth": "0", "minHeight": "0",
           "overflowY": "auto", "overflowX": "hidden"},
),
        ]),   # /main-row




    ],
)


@app.callback(
    Output("dataset-dropdown", "options"),
    [Input("dm-refresh", "data"), Input("mode-select", "value")],
)
def refresh_dataset_options(_token, _mode):
    """
    Keep the dataset picker in step with what is actually indexed.

    The options were computed once, while the layout was being built, and never
    again -- so a dataset you had just finished indexing did not appear in the
    dropdown until the app was restarted. It showed up under "Your datasets" in
    the manager, which made it look as though indexing had half-failed.

    dm-refresh is bumped by the datasets panel whenever a job finishes, which is
    exactly when the set of datasets can have changed; the mode input covers
    switching back from the manager after a relocation.
    """
    return get_matching_datasets()


@app.callback(
    [
        Output("search-box", "style"),
        Output("num-images", "style"),
        Output("story-box", "style"),
        Output("main-action-btn", "children"),
        Output("controls-bar", "style"),
        Output("moodboard-controls", "style"),
        Output("moodboard-section", "style"),
        Output("scatter-wrapper", "style"),
        Output("left-column", "style"),
        Output("right-column", "style"),
        Output("bulk-selection-btns", "style"),
        Output("save-actions-section", "style"),
        Output("datasets-panel", "style"),
        Output("main-row", "style"),
        Output("moodboard-rail", "style"),
        Output("similarity-card", "style"),
        Output("transfer-card", "style"),
        Output("image-display", "style"),
        Output("transfer-preview", "style"),
    ],
    [Input("mode-select", "value"), Input("moodboard-tool", "value")],
)
def toggle_inputs(mode, tool):
    controls_visible = {"display": "flex", "alignItems": "center", "marginTop": "10px"}
    controls_hidden = {"display": "none"}
    moodboard_controls_visible = {"display": "block", "marginTop": "10px", "padding": "10px", "backgroundColor": "#1a1a1a", "borderRadius": "5px"}
    # The results bar. This callback owns moodboard-section's style, so the
    # sticky geometry has to be stated here too -- the style set in the layout
    # is overwritten on every mode change.
    moodboard_section_visible = {"display": "block", "position": "sticky", "top": "0",
                                 "zIndex": 5, "margin": "0 -12px 12px",
                                 "padding": "9px 12px", "backgroundColor": _ui.BG,
                                 "borderBottom": f"2px solid {_ui.ACCENT}"}
    moodboard_section_hidden = {"display": "none"}
    
    # Column styles
    # Flex, not inline-block percentages: a flex child with min-height:0 can
    # shrink to fit the window, which is what keeps the app on one screen.
    left_col_normal = {"flex": "1 1 58%", "minWidth": "0", "minHeight": "0",
                       "display": "flex", "flexDirection": "column"}
    right_col_normal = {"flex": "0 0 40%", "minWidth": "0", "minHeight": "0",
                        "overflowY": "auto", "overflowX": "hidden"}
    
    # Moodboard is three bands reading materials -> work -> output: the 240px
    # rail (above), the bench, and the stage. The bench is narrower than the old
    # 31% because the rail now carries the collection.
    left_col_moodboard = {"flex": "0 0 26%", "minWidth": "0", "minHeight": "0",
                          "overflowY": "auto", "overflowX": "hidden"}
    # The 12px side padding is what the sticky results bar bleeds back through
    # with its negative margins, so the bar spans the full stage width.
    right_col_moodboard = {"flex": "1 1 auto", "minWidth": "0", "minHeight": "0",
                           "overflowY": "auto", "overflowX": "hidden",
                           "padding": "0 12px"}
    
    scatter_visible = {"flex": "1", "minHeight": "0", "position": "relative"}
    scatter_hidden = {"display": "none"}
    
    bulk_btns_visible = {"display": "flex", "gap": "8px", "marginTop": "8px"}
    save_section_visible = {"display": "block"}

    main_row_visible = {"flex": "1", "minHeight": "0", "display": "flex", "gap": "18px"}
    main_row_hidden = {"display": "none"}

    # The rail exists only in moodboard mode. Fixed width, full height, its own
    # flex column so the gallery inside it is the single flexible item.
    rail_visible = {"display": "flex", "flexDirection": "column",
                    "flex": "0 0 240px", "minHeight": "0",
                    "backgroundColor": _ui.BG, "border": f"1px solid {_ui.LINE}",
                    "borderRadius": "10px", "padding": "10px",
                    "boxSizing": "border-box"}
    rail_hidden = {"display": "none"}

    # Which tool is mounted in the bench, and therefore what the stage shows.
    hidden = {"display": "none"}
    sim_card = {"padding": "14px", "backgroundColor": "#0a1a1a",
                "borderRadius": "8px", "border": "1px solid #2a4a4a"}
    xfer_card = {"padding": "14px", "backgroundColor": "#1a1a2a",
                 "borderRadius": "8px", "border": "1px solid #4a3a5a"}
    results_list = {"overflowX": "hidden"}
    transfer_stage = {"display": "block", "overflowX": "hidden"}
    
    if mode == "prompt":
        return (
            {"display": "block", "width": "60%", "marginRight": "10px"},
            {"display": "block", "width": "15%", "marginRight": "10px"},
            {"display": "none"},
            "Search",
            controls_visible,
            controls_hidden,
            moodboard_section_hidden,
            scatter_visible,
            left_col_normal,
            right_col_normal,
            bulk_btns_visible,
            save_section_visible,
            {"display": "none"},
            main_row_visible,
            rail_hidden,
            hidden,
            hidden,
            results_list,
            hidden,
        )
    elif mode == "story":
        return (
            {"display": "none"},
            {"display": "none"},
            {"display": "block", "width": "70%", "height": "70px", "marginRight": "10px"},
            "Generate Story",
            controls_visible,
            controls_hidden,
            moodboard_section_hidden,
            scatter_visible,
            left_col_normal,
            right_col_normal,
            bulk_btns_visible,
            save_section_visible,
            {"display": "none"},
            main_row_visible,
            rail_hidden,
            hidden,
            hidden,
            results_list,
            hidden,
        )
    elif mode == "moodboard":
        return (
            {"display": "none"},
            {"display": "none"},
            {"display": "none"},
            "Search",
            controls_hidden,
            moodboard_controls_visible,
            moodboard_section_visible if tool != "transfer" else moodboard_section_hidden,
            scatter_hidden,
            left_col_moodboard,
            right_col_moodboard,
            {"display": "none"},  # hide bulk selection
            {"display": "none"},  # hide save actions
            {"display": "none"},  # hide the dataset manager
            main_row_visible,
            rail_visible,
            sim_card if tool != "transfer" else hidden,
            xfer_card if tool == "transfer" else hidden,
            hidden if tool == "transfer" else results_list,
            transfer_stage if tool == "transfer" else hidden,
        )
    else:  # datasets
        # The manager owns the whole width: there is no scatter to show, and
        # nothing here searches.
        return (
            {"display": "none"},
            {"display": "none"},
            {"display": "none"},
            "Search",
            controls_hidden,
            controls_hidden,
            moodboard_section_hidden,
            scatter_hidden,
            {"display": "none"},
            {"display": "none"},
            {"display": "none"},
            {"display": "none"},
            # This panel is a child of the root overflow:hidden flex column, so
            # it has to claim the leftover height and scroll itself -- with a
            # bare display:block its dataset list was clipped, not scrollable.
            {"display": "block", "flex": "1", "minHeight": "0",
             "overflowY": "auto", "overflowX": "hidden"},
            main_row_hidden,
            rail_hidden,
            hidden,
            hidden,
            hidden,
            hidden,
        )


# ─────────────────────────────────────────────────────────────────────────────
# MOODBOARD CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────


def _persist_upload(contents: str, filename: str | None) -> str | None:
    """
    Write a browser upload to disk and return its path.

    Content-addressed, so dropping the same picture twice does not litter the
    directory with duplicates. Registering the root matters: the media
    endpoints only serve allowlisted paths, and without it an uploaded image
    renders as a broken thumbnail.
    """
    if not contents:
        return None
    try:
        _content_type, content_string = contents.split(",", 1)
    except ValueError:
        return None
    decoded = base64.b64decode(content_string)

    ext = os.path.splitext(filename)[1] if filename else ".jpg"
    if not ext:
        ext = ".jpg"

    output_dir = _paths.ensure_dir(os.path.join(OUTPUT_DIR, "_external_refs"))
    register_media_root(OUTPUT_DIR)

    content_hash = hashlib.md5(decoded).hexdigest()[:12]
    temp_path = os.path.join(output_dir, f"ext_{content_hash}{ext}")
    if not os.path.exists(temp_path):
        with open(temp_path, "wb") as fh:
            fh.write(decoded)
    return temp_path


@app.callback(
    Output("moodboard-store", "data"),
    [
        Input({"type": "add-to-moodboard", "index": ALL}, "n_clicks"),
        Input({"type": "remove-from-moodboard", "index": ALL}, "n_clicks"),
        Input("clear-moodboard-btn", "n_clicks"),
        Input("moodboard-external-upload", "contents"),
    ],
    [State("moodboard-store", "data"),
     State("moodboard-external-upload", "filename")],
    prevent_initial_call=True,
)
def update_moodboard(add_clicks, remove_clicks, clear_clicks, upload_contents,
                     current_moodboard, upload_filename):
    """
    Add or remove images from the collection.

    The drop zone writes here and nowhere else. There used to be two upload
    zones -- one in the Similarity Search card, one by the gallery -- and BOTH
    wrote to selected-moodboard-image, so dropping a picture to search with it
    silently also made it the colour-transfer palette source. Now there is one
    intake for the whole page and it does one thing: the picture joins the
    collection. Giving it a job is a separate, deliberate click on [R] or [T].
    """
    current_moodboard = current_moodboard or []
    
    # Use ctx.triggered to get actual trigger info including value
    if not ctx.triggered:
        return dash.no_update
    
    trigger_info = ctx.triggered[0]
    trigger_prop = trigger_info.get("prop_id", "")
    trigger_value = trigger_info.get("value")
    
    # Only process if it was an actual click (value > 0)
    if trigger_value is None or trigger_value == 0:
        return dash.no_update
    
    if "clear-moodboard-btn" in trigger_prop:
        return []

    if "moodboard-external-upload" in trigger_prop and upload_contents:
        path = _persist_upload(upload_contents, upload_filename)
        if path and path not in current_moodboard:
            current_moodboard.append(path)
        return current_moodboard

    triggered_id = ctx.triggered_id
    if isinstance(triggered_id, dict):
        path = triggered_id["index"]
        if triggered_id.get("type") == "add-to-moodboard":
            if path and path not in current_moodboard:
                current_moodboard.append(path)
        elif triggered_id.get("type") == "remove-from-moodboard":
            if path in current_moodboard:
                current_moodboard.remove(path)
    
    return current_moodboard


@app.callback(
    Output("moodboard-added-notification", "children"),
    Input({"type": "add-to-moodboard", "index": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def show_moodboard_added_notification(clicks):
    """Show brief feedback when an image is added to moodboard."""
    triggered = ctx.triggered_id
    if isinstance(triggered, dict) and triggered.get("type") == "add-to-moodboard":
        # Check if this was an actual click (not initial load)
        if any(c and c > 0 for c in clicks if c is not None):
            return html.Div(
                "Image added to moodboard",
                style={
                    "color": "#4CAF50", "fontSize": "12px", "padding": "4px 8px",
                    "backgroundColor": "rgba(76, 175, 80, 0.15)", "borderRadius": "4px",
                    "display": "inline-block",
                },
            )
    return ""


@app.callback(
    Output("moodboard-ref-display", "children"),
    Output("moodboard-ref-display", "style"),
    [
        Input("selected-moodboard-image", "data"),
        Input("selected-target-image", "data"),
        Input("palette-n-colors", "value"),
        Input("mode-select", "value"),
    ],
)
def update_moodboard_ref_display(ref_image, target_image, n_colors, mode):
    """Update the Reference and Target image display in the left column."""
    # Only show in moodboard mode
    if mode != "moodboard" or (not ref_image and not target_image):
        return [], {"display": "none"}
    
    n_colors = n_colors or 16
    
    def make_image_card(image_path, label, color, is_ref=True):
        if not image_path:
            return html.Div(
                [
                    html.Div(label, style={"color": color, "fontWeight": "600", "fontSize": "11px", "marginBottom": "8px", "textTransform": "uppercase", "letterSpacing": "1px"}),
                    html.Div("Not selected", style={"color": "#666", "fontStyle": "italic", "fontSize": "12px", "padding": "30px 0", "textAlign": "center", "border": f"2px dashed {color}40", "borderRadius": "6px"}),
                ],
                style={"flex": "1", "minWidth": "120px"},
            )
        
        qpath = urllib.parse.quote(image_path)
        return html.Div(
            [
                html.Div(label, style={"color": color, "fontWeight": "600", "fontSize": "11px", "marginBottom": "8px", "textTransform": "uppercase", "letterSpacing": "1px"}),
                html.Img(
                    src=f"/preview?p={qpath}&w=300",
                    style={"width": "100%", "borderRadius": "6px", "border": f"2px solid {color}", "marginBottom": "6px"},
                ),
                html.Img(
                    src=f"/palette?p={qpath}&n={n_colors}&w=200&h=16",
                    style={"width": "100%", "borderRadius": "3px"},
                ),
                html.Div(
                    os.path.basename(image_path),
                    style={"color": "#888", "fontSize": "9px", "marginTop": "6px", "wordBreak": "break-all", "textAlign": "center"},
                ),
            ],
            style={"flex": "1", "minWidth": "120px"},
        )
    
    content = [
        html.Div(
            [
                make_image_card(ref_image, "🎨 REFERENCE", "#00bcd4", is_ref=True),
                html.Div(style={"width": "12px"}),  # Spacer
                make_image_card(target_image, "🎯 TARGET", "#e040fb", is_ref=False),
            ],
            style={"display": "flex", "gap": "0", "alignItems": "flex-start"},
        ),
    ]
    
    style = {
        "display": "block", 
        "marginTop": "12px", 
        "padding": "12px", 
        "backgroundColor": "#0a1520", 
        "borderRadius": "10px", 
        "border": "1px solid #2a3a4a",
        "position": "sticky",
        "top": "10px",
    }
    
    return content, style


@app.callback(
    Output("moodboard-gallery", "children"),
    [
        Input("moodboard-store", "data"),
        Input("selected-moodboard-image", "data"),
        Input("selected-target-image", "data"),
    ],
)
def render_moodboard_gallery(moodboard, ref_path, target_path):
    """Render clickable thumbnails in the moodboard gallery with [R] and [T] toggle badges."""
    moodboard = moodboard or []
    if not moodboard:
        # gridColumn spans the whole track list: the gallery is a grid of ~100px
        # columns, so without this the sentence wraps to one word per line.
        return [html.Div(
            [html.Div("Nothing here yet.",
                      style={"color": _ui.INK, "fontWeight": "600",
                             "marginBottom": "4px"}),
             html.Div("Search for images, then use “+ Moodboard” to collect "
                      "the ones you want to work from.",
                      style={"color": _ui.INK_DIM})],
            style={"gridColumn": "1 / -1", "padding": "26px 20px",
                   "textAlign": "center", "fontSize": "12.5px",
                   "lineHeight": "1.5"})]
    
    thumbnails = []
    for path in moodboard:
        qpath = urllib.parse.quote(path)
        is_ref = (path == ref_path)
        is_target = (path == target_path)
        
        # Border style based on role
        if is_ref and is_target:
            border = "3px solid"
            border_image = "linear-gradient(135deg, #00bcd4, #e040fb) 1"
        elif is_ref:
            border = "3px solid #00bcd4"
            border_image = None
        elif is_target:
            border = "3px solid #e040fb"
            border_image = None
        else:
            border = "2px solid #3a4a5a"
            border_image = None
        
        img_style = {
            "width": "90px", "height": "90px", "objectFit": "cover", 
            "borderRadius": "6px", "border": border, "display": "block",
        }
        if border_image:
            img_style["borderImage"] = border_image
        
        thumbnails.append(
            html.Div([
                # Image
                html.Img(
                    src=f"/thumb?p={qpath}",
                    style=img_style,
                ),
                # [R] badge - top left
                html.Button(
                    "R",
                    id={"type": "set-ref-badge", "index": path},
                    n_clicks=0,
                    style={
                        "position": "absolute", "top": "4px", "left": "4px",
                        "width": "22px", "height": "22px", "borderRadius": "4px",
                        "backgroundColor": _ui.ROLE_REF if is_ref else _ui.ROLE_OFF,
                        "color": "#fff" if is_ref else "#7a8a9a", 
                        "border": "1px solid #00bcd4" if is_ref else "1px solid #4a5a6a",
                        "fontSize": "11px", "fontWeight": "bold", "cursor": "pointer",
                        "padding": "0", "lineHeight": "20px", "textAlign": "center",
                    },
                ),
                # [T] badge - top right
                html.Button(
                    "T",
                    id={"type": "set-target-badge", "index": path},
                    n_clicks=0,
                    style={
                        "position": "absolute", "top": "4px", "right": "4px",
                        "width": "22px", "height": "22px", "borderRadius": "4px",
                        "backgroundColor": _ui.ROLE_TARGET if is_target else _ui.ROLE_OFF,
                        "color": "#fff" if is_target else "#7a8a9a",
                        "border": "1px solid #e040fb" if is_target else "1px solid #4a5a6a",
                        "fontSize": "11px", "fontWeight": "bold", "cursor": "pointer",
                        "padding": "0", "lineHeight": "20px", "textAlign": "center",
                    },
                ),
                # X button to remove - bottom right
                html.Button(
                    "×",
                    id={"type": "remove-from-moodboard", "index": path},
                    n_clicks=0,
                    style={
                        "position": "absolute", "bottom": "4px", "right": "4px",
                        "width": "18px", "height": "18px", "borderRadius": "50%",
                        "backgroundColor": "#ff4444", "color": "white", "border": "none",
                        "fontSize": "12px", "lineHeight": "1", "cursor": "pointer",
                        "padding": "0", "fontWeight": "bold", "opacity": "0.7",
                    },
                ),
            ], style={"position": "relative", "display": "inline-block"})
        )
    return thumbnails


@app.callback(
    Output("selected-moodboard-image", "data"),
    [Input({"type": "set-ref-badge", "index": ALL}, "n_clicks"),
     Input({"type": "step-to", "index": ALL}, "n_clicks")],
    State("selected-moodboard-image", "data"),
    prevent_initial_call=True,
)
def select_moodboard_ref(clicks, step_clicks, current_ref):
    """
    Set the Reference image from an [R] badge click.

    Uploads used to land here too, from two different drop zones, which is what
    made dropping a picture have three effects at once. Uploads now go to the
    collection (update_moodboard) and nothing else; this callback has exactly
    one trigger.
    """
    triggered = ctx.triggered_id
    
    # When gallery re-renders, buttons are recreated with n_clicks=0, firing this callback
    # Only proceed if an actual click happened (value > 0 in triggered)
    if not ctx.triggered:
        return dash.no_update
    
    # Check if this was a real click or just component recreation
    triggered_prop = ctx.triggered[0].get("prop_id", "")
    triggered_value = ctx.triggered[0].get("value")
    
    # A "Step here" click always sets, never toggles: the point is to move to
    # that picture, and toggling it off would leave the search with no
    # reference at all.
    if isinstance(triggered, dict) and triggered.get("type") == "step-to":
        if triggered_value is None or triggered_value == 0:
            return dash.no_update
        return triggered["index"]

    # Handle [R] badge click - toggle behavior
    if isinstance(triggered, dict) and triggered.get("type") == "set-ref-badge":
        # Only act on actual clicks (value > 0), not component recreation
        if triggered_value is None or triggered_value == 0:
            return dash.no_update
        clicked_path = triggered["index"]
        # Toggle: if already ref, unset it
        if clicked_path == current_ref:
            return None
        return clicked_path
    
    return dash.no_update


@app.callback(
    Output("selected-target-image", "data"),
    Input({"type": "set-target-badge", "index": ALL}, "n_clicks"),
    State("selected-target-image", "data"),
    prevent_initial_call=True,
)
def select_moodboard_target(clicks, current_target):
    """Set Target image from [T] badge click."""
    triggered = ctx.triggered_id
    
    # Only proceed if an actual click happened
    if not ctx.triggered:
        return dash.no_update
    
    triggered_value = ctx.triggered[0].get("value")
    
    if isinstance(triggered, dict) and triggered.get("type") == "set-target-badge":
        # Only act on actual clicks (value > 0), not component recreation
        if triggered_value is None or triggered_value == 0:
            return dash.no_update
        clicked_path = triggered["index"]
        # Toggle: if already target, unset it
        if clicked_path == current_target:
            return None
        return clicked_path
    
    return dash.no_update


def _board_options():
    """Saved boards, newest first, labelled with what is in them."""
    opts = []
    for b in _boards.listing():
        bits = f"{b['count']} image" + ("s" if b["count"] != 1 else "")
        if b["missing"]:
            bits += f", {b['missing']} missing"
        opts.append({"label": f"{b['name']}  ·  {bits}", "value": b["slug"]})
    return opts


@app.callback(
    Output("board-picker", "options"),
    [Input("mode-select", "value"), Input("board-status", "children")],
)
def refresh_board_list(mode, _status):
    """
    Rebuild the picker whenever a board is written, deleted, or the tab opens.

    board-status is an Input rather than a State on purpose: every action that
    changes the set of boards writes to it, so this needs no separate signal.
    """
    return _board_options()


@app.callback(
    [Output("board-status", "children"),
     Output("board-picker", "value", allow_duplicate=True)],
    Input("board-save-btn", "n_clicks"),
    [State("board-name", "value"),
     State("moodboard-store", "data"),
     State("selected-moodboard-image", "data"),
     State("selected-target-image", "data"),
     State("dataset-dropdown", "value")],
    prevent_initial_call=True,
)
def save_board(n, name, items, reference, target, dataset):
    """Write the collection, plus which image is [R] and which is [T]."""
    if not n:
        return dash.no_update, dash.no_update
    if not (name or "").strip():
        return "Give the board a name first.", dash.no_update
    if not items:
        return "Nothing in the collection to save.", dash.no_update
    try:
        rec = _boards.save((name or "").strip(), list(items),
                           reference=reference or "", transfer=target or "",
                           dataset=(dataset or ""))
    except Exception as e:
        return f"Could not save: {e}", dash.no_update
    n_items = len(rec["items"])
    return (f"Saved “{rec['name']}” — {n_items} image"
            + ("s" if n_items != 1 else "") + ".", rec["slug"])


@app.callback(
    [Output("moodboard-store", "data", allow_duplicate=True),
     Output("selected-moodboard-image", "data", allow_duplicate=True),
     Output("selected-target-image", "data", allow_duplicate=True),
     Output("board-name", "value"),
     Output("board-status", "children", allow_duplicate=True)],
    Input("board-load-btn", "n_clicks"),
    State("board-picker", "value"),
    prevent_initial_call=True,
)
def load_board(n, slug):
    """
    Restore a board, and say plainly if any of its files have gone.

    A board records paths, so moving or deleting originals leaves holes.
    Loading the pictures that are still there and naming the count that is not
    beats dropping them silently.
    """
    if not n or not slug:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
    rec = _boards.load(slug)
    if not rec:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, "That board could not be read."

    here, gone = _boards.present(rec)
    ref = rec.get("reference") or ""
    tgt = rec.get("transfer") or ""
    if ref and ref not in here:
        ref = ""
    if tgt and tgt not in here:
        tgt = ""

    msg = f"Loaded “{rec['name']}” — {len(here)} image" + ("s" if len(here) != 1 else "")
    if gone:
        msg += (f". {len(gone)} could not be found; "
                "run arcana-relocate if the files moved.")
    else:
        msg += "."
    return here, ref, tgt, rec.get("name", ""), msg


@app.callback(
    [Output("board-status", "children", allow_duplicate=True),
     Output("board-picker", "value", allow_duplicate=True)],
    Input("board-delete-btn", "n_clicks"),
    State("board-picker", "value"),
    prevent_initial_call=True,
)
def delete_board(n, slug):
    """Delete a saved board. The collection on screen is left alone."""
    if not n or not slug:
        return dash.no_update, dash.no_update
    rec = _boards.load(slug)
    label = rec.get("name", slug) if rec else slug
    if _boards.delete(slug):
        return f"Deleted “{label}”.", None
    return f"Could not delete “{label}”.", dash.no_update


@app.callback(
    Output("moodboard-save-confirmation", "children"),
    Input("save-moodboard-btn", "n_clicks"),
    [
        State("moodboard-store", "data"),
        State("moodboard-save-folder", "value"),
    ],
    prevent_initial_call=True,
)
def save_moodboard_images(n_clicks, moodboard, folder_name):
    """Save all moodboard reference images to a folder."""
    if not n_clicks or not moodboard:
        return dash.no_update
    
    if not folder_name or not folder_name.strip():
        return html.Span("Please enter a folder name", style={"color": "#ffcc00"})
    
    folder_name = folder_name.strip()
    # Create output path
    output_dir = _safe_output_dir("moodboards", folder_name)
    os.makedirs(output_dir, exist_ok=True)
    
    saved = 0
    for path in moodboard:
        if os.path.exists(path):
            try:
                dst = os.path.join(output_dir, os.path.basename(path))
                import shutil
                shutil.copy2(path, dst)
                saved += 1
            except Exception as e:
                print(f"Failed to save {path}: {e}")
    
    return html.Span(f"Saved {saved} images to {output_dir}", style={"color": "#4CAF50"})


@app.callback(
    Output({"type": "select-image", "index": ALL}, "on", allow_duplicate=True),
    [
        Input("moodboard-select-all", "n_clicks"),
        Input("moodboard-clear-all", "n_clicks"),
    ],
    State({"type": "select-image", "index": ALL}, "on"),
    prevent_initial_call=True,
)
def moodboard_toggle_all_selections(select_clicks, clear_clicks, current_states):
    """Select or clear all images in moodboard results."""
    triggered = ctx.triggered_id
    if triggered == "moodboard-select-all":
        return [True] * len(current_states)
    elif triggered == "moodboard-clear-all":
        return [False] * len(current_states)
    return dash.no_update


@app.callback(
    Output("moodboard-results-confirmation", "children"),
    Input("moodboard-save-selected", "n_clicks"),
    [
        State({"type": "select-image", "index": ALL}, "on"),
        State({"type": "select-image", "index": ALL}, "id"),
        State("moodboard-results-folder", "value"),
    ],
    prevent_initial_call=True,
)
def save_moodboard_selected_results(n_clicks, selections, ids, folder_name):
    """Save selected result images from moodboard search."""
    if not n_clicks:
        return dash.no_update
    
    if not folder_name or not folder_name.strip():
        return html.Span("Please enter a folder name", style={"color": "#ffcc00"})
    
    folder_name = folder_name.strip()
    output_dir = _safe_output_dir("selections", folder_name)
    os.makedirs(output_dir, exist_ok=True)
    
    saved = 0
    for sel, id_obj in zip(selections, ids):
        if sel and isinstance(id_obj, dict):
            path = id_obj.get("index", "")
            if path and os.path.exists(path) and not path.startswith("group::"):
                try:
                    dst = os.path.join(output_dir, os.path.basename(path))
                    import shutil
                    shutil.copy2(path, dst)
                    saved += 1
                except Exception as e:
                    print(f"Failed to save {path}: {e}")
    
    if saved > 0:
        return html.Span(f"Saved {saved} images to {output_dir}", style={"color": "#4CAF50"})
    else:
        return html.Span("No images selected", style={"color": "#ffcc00"})


@app.callback(
    [
        Output("image-display", "children", allow_duplicate=True),
        Output("grouped-results", "data", allow_duplicate=True),
        Output("carousel-state", "data", allow_duplicate=True),
        Output("carousel-order", "data", allow_duplicate=True),
    ],
    [Input("moodboard-search-btn", "n_clicks"),
     Input({"type": "step-to", "index": ALL}, "n_clicks")],
    [
        State("selected-moodboard-image", "data"),
        State("moodboard-palette-check", "value"),
        State("palette-method", "value"),
        State("moodboard-style-check", "value"),
        State("style-method", "value"),
        State("palette-n-colors", "value"),
        State("show-palette-swatches", "value"),
        State("moodboard-prompt", "value"),
        State("moodboard-num", "value"),
        State("moodboard-img-size", "value"),
        State("moodboard-columns", "value"),
        State("dataset-dropdown", "value"),
        State("group-similar", "on"),
        State("sim-thresh", "value"),
    ],
    prevent_initial_call=True,
)
def moodboard_similarity_search(n_clicks, step_clicks, ref_image, use_palette, palette_method, use_style, style_method, 
                                 n_colors, show_swatches, prompt, num_results, img_size, columns, dataset_value, group_on, sim_thresh):
    """Search for similar images using palette/style, optionally constrained by prompt."""
    # A "Step here" click carries the new reference in its own id, so this
    # does not wait for selected-moodboard-image to come back from the
    # browser. Both callbacks fire from the same click and their order is
    # not guaranteed; reading the trigger removes the question.
    trig = ctx.triggered_id
    stepped = isinstance(trig, dict) and trig.get("type") == "step-to"
    if stepped:
        fired = ctx.triggered[0].get("value")
        if fired is None or fired == 0:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update
        ref_image = trig.get("index") or ref_image
    elif not n_clicks:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update

    if not ref_image or not dataset_value:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    if not PALETTE_STYLE_AVAILABLE:
        return [html.Div("Palette/style features not available. Run build with --features palette,style", 
                        style={"color": "#ff6666", "padding": "10px"})], [], {}, []
    
    if not use_palette and not use_style:
        return [html.Div("Please enable at least one feature (Color Palette or Style/Texture)", 
                        style={"color": "#ffcc00", "padding": "10px"})], [], {}, []
    
    # Parse dataset
    try:
        parts = dataset_value.split("::")
        db_name = parts[0]
        modality = parts[2] if len(parts) == 3 else "image"
        
        # Preload palette cache for this db (speeds up palette rendering)
        if db_name not in _palette_cache:
            with _palette_cache_lock:
                if db_name not in _palette_cache:
                    _palette_cache[db_name] = _load_palette_cache(db_name)
    except:
        return [html.Div("Invalid dataset", style={"color": "#ff6666"})], [], {}, []
    
    if modality != "image":
        return [html.Div("Similarity search only available for images", style={"color": "#ff6666"})], [], {}, []
    
    # Load index
    try:
        index, idx2path = load_index(db_name, modality=modality)
    except Exception as e:
        return [html.Div(f"Failed to load index: {e}", style={"color": "#ff6666"})], [], {}, []
    
    num_results = num_results or 50
    n_colors = n_colors or 16
    palette_method = palette_method or "emd"
    style_method = style_method or "gram"
    img_size = img_size or "medium"
    columns = columns or 2
    
    # Image size settings
    size_config = {
        "small": {"w": 300, "srcset": False},
        "medium": {"w": 600, "srcset": True},
        "large": {"w": 900, "srcset": True},
        "full": {"w": 1400, "srcset": True},
    }
    cfg = size_config.get(img_size, size_config["medium"])
    
    # Step 1: If prompt provided, first filter by CLIP similarity
    candidate_paths = None
    if prompt and prompt.strip():
        # Use the local search function (CLIP-based)
        clip_results = search(index, idx2path, prompt.strip(), min(num_results * 3, len(idx2path)), modality=modality)
        candidate_paths = {r[1] for r in clip_results}  # Set of paths
    
    # Step 2: Compute palette/style similarity
    try:
        scores = {}  # path -> combined score
        
        if use_palette:
            palette_results = search_by_palette(ref_image, db_name, idx2path, method=palette_method, n_colors=n_colors, top_k=len(idx2path))
            for path, score in palette_results:
                if candidate_paths is None or path in candidate_paths:
                    weight = 0.5 if use_style else 1.0  # Only split weight if using both features
                    scores[path] = scores.get(path, 0) + score * weight
        
        if use_style:
            # Try specified method, fall back to edge
            try:
                style_results = search_by_style(ref_image, db_name, idx2path, method=style_method, top_k=len(idx2path))
            except:
                style_results = search_by_style(ref_image, db_name, idx2path, method="edge", top_k=len(idx2path))
            
            for path, score in style_results:
                if candidate_paths is None or path in candidate_paths:
                    weight = 0.5 if use_palette else 1.0
                    scores[path] = scores.get(path, 0) + score * weight
        
        # Sort by combined score
        results = sorted(scores.items(), key=lambda x: -x[1])[:num_results]
        
    except FileNotFoundError as e:
        return [html.Div(f"Features not found: {e}. Run build with --features palette,style", 
                        style={"color": "#ff6666", "padding": "10px"})], [], {}, []
    except Exception as e:
        return [html.Div(f"Search error: {e}", style={"color": "#ff6666", "padding": "10px"})], [], {}, []
    
    if not results:
        return [html.Div("No results found", style={"color": "#ffcc00", "padding": "10px"})], [], {}, []
    
    show_palettes = show_swatches and "show" in show_swatches
    
    # Filter out reference image from results
    filtered_results = [(p, s) for p, s in results if os.path.normpath(p) != os.path.normpath(ref_image)]
    
    # Group twins if enabled
    if group_on and len(filtered_results) > 1:
        # Build paths and scores dict
        paths = [p for p, s in filtered_results]
        score_map = {p: s for p, s in filtered_results}
        
        # Get keys from idx2path (reverse mapping)
        path2key = {v: k for k, v in idx2path.items()}
        keys = []
        valid_paths = []
        for p in paths:
            if p in path2key:
                keys.append(path2key[p])
                valid_paths.append(p)
        
        if keys:
            groups = _cosine_group(keys, valid_paths, index, float(sim_thresh or 0.08))
        else:
            groups = [{"gid": f"g{i}", "keys": [], "paths": [p]} for i, (p, s) in enumerate(filtered_results)]
    else:
        # No grouping - each result is its own group
        groups = [{"gid": f"g{i}", "keys": [], "paths": [p]} for i, (p, s) in enumerate(filtered_results)]
        score_map = {p: s for p, s in filtered_results}
    
    # Build result grid
    result_items = []
    rank = 0
    
    for g in groups:
        n = len(g["paths"])
        first = g["paths"][0]
        first_score = score_map.get(first, 0)
        qpath = urllib.parse.quote(first)
        rank += 1
        
        if n == 1:
            # Single image card
            if cfg["srcset"]:
                result_img = html.Img(
                    src=f"/preview?p={qpath}&w={cfg['w']}",
                    srcSet=f"/preview?p={qpath}&w=600 600w, /preview?p={qpath}&w=900 900w, /preview?p={qpath}&w=1400 1400w",
                    sizes="(max-width: 900px) 90vw, 50vw",
                    style={"width": "100%", "borderRadius": "6px", "cursor": "pointer"},
                )
            else:
                result_img = html.Img(
                    src=f"/preview?p={qpath}&w={cfg['w']}",
                    style={"width": "100%", "borderRadius": "6px", "cursor": "pointer"},
                )
            
            result_elements = [
                html.Div(
                    f"#{rank} \u2022 {first_score:.2f}", 
                    style={"color": "#888", "fontSize": "11px", "marginBottom": "6px"}
                ),
                result_img,
            ]
            if show_palettes:
                result_elements.append(
                    html.Img(
                        src=f"/palette?p={qpath}&n={n_colors}&w={cfg['w']}&h=20&db={db_name}",
                        style={"width": "100%", "marginTop": "6px", "borderRadius": "4px"},
                    )
                )
            result_elements.append(
                html.Div([
                    daq.BooleanSwitch(id={"type": "select-image", "index": first}, on=False),
                    html.Button("+ Moodboard", id={"type": "add-to-moodboard", "index": first}, 
                               n_clicks=0, style={"marginLeft": "8px", "fontSize": "11px", "padding": "3px 8px", "backgroundColor": "#333", "border": "none", "borderRadius": "3px", "color": "#aaa", "cursor": "pointer"}),
                                    # Finding a picture is a walk: you land near it and then
                    # step. Stepping used to mean adding the image to the
                    # collection, switching to the Moodboard, clicking its [R]
                    # badge and hitting Find Similar again -- four actions to
                    # move one notch. This does the whole step.
                    html.Button("Step here",
                                id={"type": "step-to", "index": first},
                                n_clicks=0,
                                title="Make this the reference and search again",
                                style={"marginLeft": "6px", "fontSize": "11px",
                                       "padding": "3px 8px",
                                       "backgroundColor": "#00bcd4", "border": "none",
                                       "borderRadius": "3px", "color": "#04222a",
                                       "fontWeight": "600", "cursor": "pointer"}),
                ], style={"display": "flex", "alignItems": "center", "marginTop": "8px"})
            )
            
            result_items.append(
                html.Div(result_elements, style={"padding": "10px", "backgroundColor": "#1e1e1e", "borderRadius": "8px", "border": "1px solid #333"})
            )
        else:
            # Carousel for twins
            media_el = html.Img(
                id={"type": "carousel-img", "gid": g["gid"]},
                src=f"/preview?p={qpath}&w={cfg['w']}",
                srcSet=f"/preview?p={qpath}&w=600 600w, /preview?p={qpath}&w=900 900w, /preview?p={qpath}&w=1400 1400w" if cfg["srcset"] else "",
                sizes="(max-width: 900px) 90vw, 50vw",
                style={"width": "100%", "display": "block", "marginBottom": "10px", "borderRadius": "5px"},
            )
            
            card_elements = [
                html.Div(
                    f"#{rank} \u2022 {first_score:.2f} \u2022 twins: {n}", 
                    style={"color": "#888", "fontSize": "11px", "marginBottom": "6px"}
                ),
                html.Div(
                    [
                        media_el,
                        html.Button("◀", id={"type": "left", "gid": g["gid"]}, n_clicks=0,
                                    style={"position": "absolute", "left": "8px", "top": "50%",
                                        "transform": "translateY(-50%)",
                                        "backgroundColor": "rgba(0,0,0,0.6)", "color": "#fff",
                                        "border": "none", "borderRadius": "9999px",
                                        "width": "36px", "height": "36px", "zIndex": 2, "cursor": "pointer"}),
                        html.Button("▶", id={"type": "right", "gid": g["gid"]}, n_clicks=0,
                                    style={"position": "absolute", "right": "8px", "top": "50%",
                                        "transform": "translateY(-50%)",
                                        "backgroundColor": "rgba(0,0,0,0.6)", "color": "#fff",
                                        "border": "none", "borderRadius": "9999px",
                                        "width": "36px", "height": "36px", "zIndex": 2, "cursor": "pointer"}),
                    ],
                    style={"position": "relative", "overflow": "hidden"},
                ),
                html.Div(id={"type": "carousel-counter", "gid": g["gid"]}, children=f"1/{n}",
                        style={"textAlign": "center", "margin": "4px 0 8px 0", "opacity": 0.8}),
            ]
            
            if show_palettes:
                card_elements.append(
                    html.Img(
                        src=f"/palette?p={qpath}&n={n_colors}&w={cfg['w']}&h=20&db={db_name}",
                        style={"width": "100%", "marginTop": "6px", "borderRadius": "4px"},
                    )
                )
            
            card_elements.append(
                html.Div([
                    daq.BooleanSwitch(id={"type": "select-image", "index": f"group::{g['gid']}"}, on=False),
                    html.Button("+ Moodboard", id={"type": "add-to-moodboard", "index": first}, 
                               n_clicks=0, style={"marginLeft": "8px", "fontSize": "11px", "padding": "3px 8px", "backgroundColor": "#333", "border": "none", "borderRadius": "3px", "color": "#aaa", "cursor": "pointer"}),
                ], style={"display": "flex", "alignItems": "center", "marginTop": "8px"})
            )
            
            result_items.append(
                html.Div(card_elements, style={"padding": "10px", "backgroundColor": "#1e1e1e", "borderRadius": "8px", "border": "1px solid #333"})
            )
    
    # Grid wrapper
    grid_style = {
        "display": "grid",
        "gridTemplateColumns": f"repeat({columns}, 1fr)",
        "gap": "12px",
    }
    
    # Build carousel state for groups with multiple items
    car_state = {g["gid"]: 0 for g in groups}
    carousel_order = [g["gid"] for g in groups if len(g.get("paths", [])) > 1]
    
    return [html.Div(result_items, style=grid_style)], groups, car_state, carousel_order


# ─────────────────────────────────────────────────────────────────
# Similarity Search Database Availability Callback
# ─────────────────────────────────────────────────────────────────

@app.callback(
    [
        Output("palette-card", "style"),
        Output("style-card", "style"),
        Output("palette-db-status", "children"),
        Output("style-db-status", "children"),
        Output("moodboard-palette-check", "value"),
        Output("moodboard-style-check", "value"),
    ],
    Input("dataset-dropdown", "value"),
    prevent_initial_call=True,
)
def update_feature_availability(dataset_value):
    """Check if palette/style databases exist for the selected dataset and update UI."""
    enabled_style = {"padding": "10px", "backgroundColor": "#252525", "borderRadius": "8px", "border": "1px solid #333", "flex": "1", "minWidth": "140px"}
    disabled_style = {"padding": "10px", "backgroundColor": "#1a1a1a", "borderRadius": "8px", "border": "1px solid #222", "flex": "1", "minWidth": "140px", "opacity": "0.5", "pointerEvents": "none"}
    
    if not dataset_value:
        return enabled_style, enabled_style, "", "", ["palette"], []
    
    # Parse dataset name
    try:
        parts = dataset_value.split("::")
        db_name = parts[0]
    except:
        return enabled_style, enabled_style, "", "", ["palette"], []
    
    # Check for feature databases
    palette_path = os.path.join(DB_DIR, f"features_{db_name}_palette.npz")
    style_path = os.path.join(DB_DIR, f"features_{db_name}_style.npz")
    
    has_palette = os.path.exists(palette_path)
    has_style = os.path.exists(style_path)
    
    # Build status indicators
    palette_status = "" if has_palette else "⚠️ No DB"
    style_status = "" if has_style else "⚠️ No DB"
    
    # Set default values - enable if available
    palette_value = ["palette"] if has_palette else []
    style_value = []  # Style unchecked by default
    
    return (
        enabled_style if has_palette else disabled_style,
        enabled_style if has_style else disabled_style,
        palette_status,
        style_status,
        palette_value,
        style_value,
    )


# ─────────────────────────────────────────────────────────────────
# Color Transfer Callbacks
# ─────────────────────────────────────────────────────────────────

# Disable ModFlows-only parameters when LAB method is selected
@app.callback(
    # A bare Output, not a one-element list: Dash matches the return value
    # against the declared shape, so a list here would require the function to
    # return a list too. It used to have three outputs; reducing it to one left
    # the brackets behind and every method change raised
    # SchemaTypeValidationError.
    Output("color-transfer-quality-wrapper", "style"),
    Input("color-transfer-method", "value"),
)
def toggle_method_params(method):
    """
    Grey out the ModFlows-only controls when LAB is selected.

    LAB (Reinhard) matches per-channel mean and standard deviation in LAB space
    in one shot: there is no working resolution, no integration steps and no
    full-res pass to configure. Strength is its only parameter.
    """
    if method == "lab":
        # LAB is a closed-form statistical transfer: it has no flow to run, so
        # a quality choice would be a control that changes nothing.
        return {"display": "flex", "alignItems": "center", "opacity": "0.4",
                "pointerEvents": "none"}
    return {"display": "flex", "alignItems": "center"}


# ─────────────────────────────────────────────────────────────────────────────
# THE MODFLOWS CHECKPOINT
# ─────────────────────────────────────────────────────────────────────────────

def _ct_model_card(job_snapshot=None):
    """One line describing whether the neural method can run, plus a way to fix it."""
    try:
        from . import color_transfer as _ct
    except ImportError:
        import color_transfer as _ct

    if job_snapshot and not job_snapshot.get("status") in (None, "done", "failed",
                                                           "cancelled"):
        pct = int(round((job_snapshot.get("fraction") or 0) * 100))
        return html.Div([
            html.Div(job_snapshot.get("message") or "Downloading...",
                     style={"fontSize": "11.5px", "color": _ui.ACCENT}),
            html.Div(style={"height": "4px", "borderRadius": "2px",
                            "backgroundColor": _ui.SURFACE_2, "marginTop": "6px"},
                     children=html.Div(style={
                         "width": f"{pct}%", "height": "100%",
                         "borderRadius": "2px", "backgroundColor": _ui.ACCENT,
                         "transition": "width .3s"})),
        ])

    if job_snapshot and job_snapshot.get("status") == "failed":
        return html.Div(job_snapshot.get("error") or "The download failed.",
                        style={"fontSize": "11.5px", "color": _ui.BAD})

    try:
        st = _ct.status()
    except Exception:
        return ""

    if st["ready"]:
        return ""                      # nothing to say when it just works
    if not st["source"]:
        return html.Div(
            "ModFlows is not installed in this build — use the LAB method.",
            style={"fontSize": "11.5px", "color": _ui.INK_DIM})
    return html.Div([
        html.Div(f"ModFlows needs a one-off {st['download_mb']} MB model download.",
                 style={"fontSize": "11.5px", "color": _ui.INK_DIM,
                        "marginBottom": "6px"}),
        html.Button("Download colour model", id="ct-download-btn", n_clicks=0,
                    style=_ui.button("secondary", padding="6px 12px",
                                     fontSize="12px")),
    ])


@app.callback(
    [Output("ct-job", "data"), Output("ct-poll", "disabled"),
     Output("ct-model-status", "children", allow_duplicate=True)],
    Input("ct-download-btn", "n_clicks"),
    prevent_initial_call=True,
)
def start_modflows_download(n):
    if not n:
        raise dash.exceptions.PreventUpdate
    try:
        from . import color_transfer as _ct
        from .jobs import MANAGER
    except ImportError:
        import color_transfer as _ct
        from jobs import MANAGER

    # One job at a time, shared with indexing and encoder downloads: they all
    # compete for the same disk and network.
    if MANAGER.active():
        return dash.no_update, dash.no_update, html.Div(
            "Something else is running — wait for it to finish.",
            style={"fontSize": "11.5px", "color": _ui.WARN})

    def job(handle):
        _ct.download_checkpoint(
            progress=lambda frac, msg: handle.update(fraction=frac, message=msg))
        handle.update(fraction=1.0, message="Colour model ready")
        return {"ok": True}

    jid = MANAGER.submit(job, kind="download", label="Colour model")
    return jid, False, _ct_model_card({"status": "running", "fraction": 0.0,
                                       "message": "Starting download..."})


@app.callback(
    [Output("ct-model-status", "children"), Output("ct-poll", "disabled",
                                                   allow_duplicate=True)],
    [Input("ct-poll", "n_intervals"), Input("mode-select", "value"),
     Input("moodboard-tool", "value")],
    State("ct-job", "data"),
    prevent_initial_call="initial_duplicate",
)
def poll_modflows_download(_ticks, mode, tool, job_id):
    try:
        from .jobs import MANAGER
    except ImportError:
        from jobs import MANAGER
    snap = MANAGER.snapshot(job_id) if job_id else None
    finished = (snap or {}).get("status") in ("done", "failed", "cancelled")
    return _ct_model_card(snap), (snap is None or finished)


# ─────────────────────────────────────────────────────────────────────────────
# WHICH COLOURS OF THE REFERENCE GET USED
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(
    [Output("ct-lightness", "value"), Output("ct-saturation", "value")],
    Input("ct-reset", "n_clicks"),
    prevent_initial_call=True,
)
def reset_reference_filter(n):
    if not n:
        raise dash.exceptions.PreventUpdate
    return [0, 100], [0, 100]


@app.callback(
    [Output("ct-swatches", "children"), Output("ct-keep-note", "children")],
    [Input("selected-moodboard-image", "data"),
     Input("ct-lightness", "value"),
     Input("ct-saturation", "value")],
)
def preview_reference_filter(ref_path, lightness, saturation):
    """
    Draw the colours the transfer would actually use.

    Runs on every slider move, so it works off a 256px sample and uniform
    quantisation rather than anything iterative.
    """
    if not ref_path:
        return [], "Pick a Reference [R] picture to choose its colours."

    l_min, l_max = (lightness or [0, 100])
    s_min, s_max = (saturation or [0, 100])

    try:
        from . import refselect
    except ImportError:
        import refselect

    try:
        resolved = resolve_path(ref_path)
        if not os.path.exists(resolved):
            return [], "The reference file is no longer on disk."
        with Image.open(resolved) as im:
            im.load()
            colours = refselect.palette_strip(im, n=14, l_min=l_min, l_max=l_max,
                                              s_min=s_min, s_max=s_max,
                                              source_path=resolved)
            _proxy, frac = refselect.filter_reference(im, l_min, l_max, s_min, s_max)
    except Exception as e:
        return [], f"Could not read the reference: {type(e).__name__}"

    if not colours:
        return [], html.Span("Nothing is left in that range — the transfer would "
                             "use the whole picture.", style={"color": _ui.WARN})

    # Widths are proportional to each colour's share, exactly like the strips
    # /palette draws under the Reference and Target thumbnails. Equal widths
    # made a 70%-black photograph look like an evenly-spread dark palette, and
    # disagreed with the strip a few centimetres below it. Proportional widths
    # also make the sliders legible: raising the lightness floor visibly hands
    # the black band's width over to the colours you actually want.
    strip = [html.Div(style={"flexGrow": max(share, 0.0005), "flexBasis": "0",
                             "backgroundColor": c},
                      title=f"{c} · {share * 100:.1f}%")
             for c, share in colours]
    pct = frac * 100
    if frac >= 1.0:
        note = "Using the whole reference — widths show each colour's share."
    elif frac < refselect.MIN_KEEP:
        note = html.Span(f"Only {pct:.2f}% of the picture survives — that is very "
                         "little to estimate from.", style={"color": _ui.WARN})
    else:
        note = f"Using {pct:.1f}% of the reference's pixels."
    return strip, note


@app.callback(
    Output("color-transfer-quality-note", "children"),
    [Input("color-transfer-quality", "value"),
     Input("color-transfer-method", "value")],
)
def describe_transfer_quality(quality, method):
    """Say what the chosen preset costs, and that the output is full size."""
    if method == "lab":
        return "LAB is instant and needs no model."
    try:
        from . import color_transfer as _ct
        from . import gpu as _g
    except ImportError:
        import color_transfer as _ct
        import gpu as _g
    p = _ct.QUALITY_PRESETS.get(quality or _ct.DEFAULT_QUALITY)
    if not p:
        return ""
    # Measured on a 6000x4000 photograph; the GPU numbers barely move between
    # presets because building and applying the LUT is CPU work.
    secs = {"quick": (5, 5), "balanced": (11, 4), "best": (38, 5)}[quality or "balanced"]
    took = secs[1] if _g.available() else secs[0]
    return f"{p['note']} — about {took}s here, output at full resolution."


@app.callback(
    # Two outputs: a one-line receipt that stays in the card, and the picture
    # itself, which goes to the stage. The preview used to be a 640px image
    # rendered width:100% inside a ~300px card in the bench -- the result of the
    # operation was the smallest thing on screen. Both tools now obey one rule:
    # you set up on the left, the result appears on the right.
    [Output("color-transfer-status", "children"),
     Output("transfer-preview", "children")],
    Input("color-transfer-btn", "n_clicks"),
    State("selected-moodboard-image", "data"),  # Reference (palette source)
    State("selected-target-image", "data"),      # Target (receives colors)
    State("color-transfer-method", "value"),
    State("color-transfer-strength", "value"),
    State("color-transfer-quality", "value"),
    State("ct-lightness", "value"),
    State("ct-saturation", "value"),
    prevent_initial_call=True,
)
def perform_color_transfer(n_clicks, ref_path, target_path, method, strength,
                           quality, ct_lightness, ct_saturation):
    """Transfer colors from reference image to target image."""
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    
    # Validate inputs
    if not ref_path:
        return html.Div("⚠️ No Reference [R] image selected", style={"color": "#f39c12"}), ""
    if not target_path:
        return html.Div("⚠️ No Target [T] image selected", style={"color": "#f39c12"}), ""
    
    # Check method availability
    if method == "modflows" and not COLOR_TRANSFER_AVAILABLE:
        return html.Div("⚠️ ModFlows not available (check installation)", style={"color": "#e74c3c"}), ""
    if method == "lab" and not LAB_TRANSFER_AVAILABLE:
        return html.Div("⚠️ LAB transfer not available", style={"color": "#e74c3c"}), ""
    
    try:
        import time
        start_time = time.time()
        
        # Resolve paths
        ref_resolved = resolve_path(ref_path)
        target_resolved = resolve_path(target_path)
        
        if not os.path.exists(ref_resolved):
            return html.Div(f"⚠️ Reference file not found", style={"color": "#e74c3c"}), ""
        if not os.path.exists(target_resolved):
            return html.Div(f"⚠️ Target file not found", style={"color": "#e74c3c"}), ""
        
        # Parse options
        strength_val = float(strength) if strength else 1.0
        
        # Load images
        content_img = Image.open(target_resolved).convert("RGB")
        style_img = Image.open(ref_resolved).convert("RGB")

        # Narrow the reference to the colours the user actually chose. Both
        # methods below read a colour *distribution* from style_img and nothing
        # else, so swapping in a proxy built from the surviving pixels steers
        # them without either method needing to know this feature exists.
        l_min, l_max = (ct_lightness or [0, 100])
        s_min, s_max = (ct_saturation or [0, 100])
        try:
            from . import refselect
        except ImportError:
            import refselect
        style_img, kept_frac = refselect.filter_reference(
            style_img, l_min, l_max, s_min, s_max)
        # An impossible range returns the original untouched rather than a blank
        # reference; say so instead of silently ignoring the sliders.
        filter_note = ""
        if kept_frac == 0.0:
            filter_note = " — colour range matched nothing, used the whole reference"
        elif kept_frac < 1.0:
            filter_note = f" — from {kept_frac * 100:.1f}% of the reference"
        
        # Perform transfer based on method
        if method == "lab":
            result_img = lab_color_transfer_pil(content_img, style_img, strength=strength_val)
            method_label = "LAB (Reinhard)"
            device_str = "CPU"
        else:
            # ModFlows. The flow runs at the preset's size; the colour
            # mapping it finds is then applied to the original picture at full
            # resolution, so the output is never smaller than the input.
            try:
                from . import color_transfer as _ct
            except ImportError:
                import color_transfer as _ct
            preset = _ct.QUALITY_PRESETS.get(quality or _ct.DEFAULT_QUALITY,
                                             _ct.QUALITY_PRESETS[_ct.DEFAULT_QUALITY])
            low = transfer_colors(
                content=content_img,
                style=style_img,
                strength=strength_val,
                steps=preset["steps"],
                max_size=preset["max_size"],
            )
            result_img = _ct.transfer_at_full_resolution(
                target_resolved, low, preset["lut"])
            method_label = f"ModFlows, {preset['label'].lower()}"
            device_info = get_device_info() if get_device_info else {"device": "unknown"}
            device_str = device_info.get("device", "unknown")
        
        elapsed = time.time() - start_time
        
        # Save to output/color_transfer/
        output_dir = _paths.ensure_dir(os.path.join(OUTPUT_DIR, "color_transfer"))

        target_name = os.path.splitext(os.path.basename(target_resolved))[0]
        ref_name = os.path.splitext(os.path.basename(ref_resolved))[0]
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        method_suffix = "lab" if method == "lab" else "mf"
        output_filename = _paths.fit_filename(
            output_dir, target_name, ref_name, f"{method_suffix}_{timestamp}", ".png"
        )
        output_path = os.path.join(output_dir, output_filename)
        # The filename has to be truncated to fit the filesystem, which loses
        # which images this actually came from. Keep the full provenance inside
        # the PNG so a result is always traceable back to its inputs.
        from PIL import PngImagePlugin
        meta = PngImagePlugin.PngInfo()
        meta.add_text("arcana:target", str(target_resolved))
        meta.add_text("arcana:reference", str(ref_resolved))
        meta.add_text("arcana:method", method_label)
        meta.add_text("arcana:strength", str(strength_val))
        meta.add_text("arcana:device", str(device_str))
        result_img.save(output_path, "PNG", pnginfo=meta)
        
        # Show the result, not just a filename. Embedded as a downscaled JPEG
        # data URI rather than a /preview URL: the output directory is not a
        # media root, and one preview does not justify making it servable.
        preview = result_img.copy()
        preview.thumbnail((640, 640), Image.LANCZOS)
        buf = BytesIO()
        preview.convert("RGB").save(buf, "JPEG", quality=88)
        preview_uri = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

        receipt = html.Div(f"✓ Done in {elapsed:.1f}s — see the panel on the right",
                           style={"color": _ui.OK, "fontSize": "11.5px"})
        stage = html.Div([
            html.Div(
                [
                    html.Span(f"TRANSFER · {method_label}{filter_note}", style={
                        "fontWeight": "700", "fontSize": "12px",
                        "letterSpacing": "1px", "color": _ui.INK}),
                    html.Span(
                        f"{result_img.size[0]}×{result_img.size[1]} · {device_str} · {elapsed:.2f}s",
                        style={"color": _ui.INK_DIM, "fontSize": "11px"}),
                ],
                style={"display": "flex", "alignItems": "baseline", "gap": "10px",
                       "flexWrap": "wrap", "margin": "0 -12px 12px",
                       "padding": "9px 12px",
                       "borderBottom": f"2px solid {_ui.ROLE_TARGET}"},
            ),
            html.Img(
                src=preview_uri,
                title=output_filename,
                style={"display": "block", "maxWidth": "100%", "maxHeight": "62vh",
                       "borderRadius": "8px", "border": f"1px solid {_ui.LINE}"},
            ),
            html.Div(f"Saved to {os.path.join('output', 'color_transfer')}",
                     style={"color": _ui.INK_FAINT, "fontSize": "11px", "marginTop": "8px"}),
            html.Div(output_filename,
                     style={"color": _ui.INK_FAINT, "fontSize": "10px",
                            "wordBreak": "break-all"}),
        ])
        return receipt, stage

    except Exception as e:
        return (html.Div(f"✗ Error: {str(e)}",
                         style={"color": "#e74c3c", "wordBreak": "break-word"}), "")


def _poetry_job(story_cache, folder, strength_val):
    """
    The diffusion pass, as a function of a progress handle.

    Kept free of Dash objects on purpose: it runs on the job manager's worker
    thread and returns plain data, which the polling callback turns into
    components. It reports once per scene and once per denoising step within a
    scene, because on the CPU a single scene is over half a minute.
    """
    def run(handle):
        import torch
        from PIL import Image as _Image

        subfolder = folder or "story"
        output_dir = _safe_output_dir("stories", subfolder, "poetry_injected")
        os.makedirs(output_dir, exist_ok=True)

        device = _gpu.device()
        # hardware_note(), not describe(): on the packaged build this is where
        # a user first meets the 37.5 s-per-scene CPU path, and it should say
        # plainly that the card in their machine is not being used.
        handle.update(fraction=0.0, message="Loading the image model",
                      detail=_gpu.hardware_note())

        from diffusers import StableDiffusionImg2ImgPipeline
        # fp16 is a CUDA thing: on CPU most fp16 kernels are unimplemented, so
        # asking for the fp16 variant made story mode raise on every machine
        # without an NVIDIA GPU rather than merely being slow.
        if device == "cuda":
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                "stabilityai/sd-turbo", torch_dtype=torch.float16, variant="fp16"
            ).to(device)
        else:
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                "stabilityai/sd-turbo", torch_dtype=torch.float32
            ).to(device)

        # xformers is CUDA-only and optional; it raises when unavailable.
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception as e:
            print(f"[INFO] xformers attention unavailable, continuing without it: {e}")
        pipe.enable_vae_slicing()
        pipe.safety_checker = None
        pipe.watermark = None

        strength = 0.72 if strength_val is None else max(0.0, min(1.0, float(strength_val)))
        num_steps = 4
        guidance_scale = 1.0
        negative_prompt = "text, letters, watermark, logo, blurry, low quality"

        scenes = story_cache["story"]
        n = len(scenes)
        out_items = []

        for idx, item in enumerate(scenes):
            handle.raise_if_cancelled()
            img_path = resolve_path(item["path"])
            prompt = item["text"]
            handle.update(done=idx, total=n,
                          message=f"Scene {idx + 1} of {n}",
                          detail=prompt[:70])

            init_img = _Image.open(img_path).convert("RGB")
            max_width = 1024
            w, h = init_img.size
            if w > max_width:
                scale = max_width / float(w)
                new_w = (max_width // 8) * 8
                new_h = (int(h * scale) // 8) * 8
            else:
                new_w, new_h = (w // 8) * 8, (h // 8) * 8
            init_img = init_img.resize((new_w, new_h), _Image.LANCZOS)

            # Within-scene reporting. Without it the bar would sit still for the
            # ~37 s a single CPU scene takes, which is the whole complaint.
            def _step_cb(pipe_ref, step, timestep, cbk, _idx=idx, _p=prompt):
                frac = (_idx + (step + 1) / float(num_steps)) / float(n)
                handle.update(fraction=frac,
                              message=f"Scene {_idx + 1} of {n}",
                              detail=f"step {step + 1}/{num_steps} · {_p[:50]}")
                return cbk

            gen = torch.manual_seed(2222 + idx)
            out_img = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=init_img,
                strength=strength,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                generator=gen,
                callback_on_step_end=_step_cb,
            ).images[0]

            gen_name = _short_poetry_name(img_path, prompt, idx, ext="png")
            poetry_img_path = os.path.join(output_dir, gen_name)
            out_img.save(_win_longpath(poetry_img_path))
            poetry_img_path = os.path.abspath(poetry_img_path)

            buffered = BytesIO()
            out_img.save(buffered, format="JPEG")
            poetry_img_str = base64.b64encode(buffered.getvalue()).decode()

            out_items.append({
                "text": prompt,
                "path": item["path"],
                "original_img_str": item.get("img_str", ""),
                "poetry_img_str": poetry_img_str,
                "poetry_img_path": poetry_img_path,
            })

        pipe.to("cpu")
        del pipe
        if device == "cuda":
            torch.cuda.empty_cache()

        handle.update(fraction=1.0, done=n, total=n, message="Done", detail="")
        return {"items": out_items, "output_dir": output_dir}

    return run


_POETRY_BOX = {"display": "none", "flexDirection": "column",
               "gap": "6px", "minWidth": "220px"}


def _poetry_bar(pct):
    return {"height": "100%", "width": f"{pct}%",
            "backgroundColor": "#00bcd4", "transition": "width 0.3s"}


@app.callback(
    [
        Output("poetry-job", "data"),
        Output("poetry-poll", "disabled"),
        Output("poetry-progress", "style"),
    ],
    Input("inject-poetry-btn", "n_clicks"),
    State("story-cache", "data"),
    State("save-folder", "value"),
    State("poetry-strength", "value"),
    prevent_initial_call=True,
)
def inject_poetry(n_clicks, story_cache, folder, strength_val):
    """
    Start the diffusion pass and hand the polling callback a job id.

    This used to do the whole thing inline and return the finished images, so
    the browser sat on one request for minutes with no output of any kind. The
    job manager already existed for indexing; this now uses it.
    """
    if not story_cache or "story" not in story_cache or not story_cache["story"]:
        return None, True, dict(_POETRY_BOX)

    job_id = jobs.MANAGER.submit(
        _poetry_job(story_cache, folder, strength_val),
        kind="poetry", label="Injecting poetry",
    )
    return ({"id": job_id, "chunks": story_cache.get("chunks")}, False,
            dict(_POETRY_BOX, display="flex"))


@app.callback(
    [
        Output("poetry-progress-msg", "children"),
        Output("poetry-progress-bar", "style"),
        Output("poetry-poll", "disabled", allow_duplicate=True),
        Output("poetry-progress", "style", allow_duplicate=True),
        Output("save-confirmation", "children", allow_duplicate=True),
        Output("story-cache", "data", allow_duplicate=True),
        Output("image-display", "children", allow_duplicate=True),
    ],
    Input("poetry-poll", "n_intervals"),
    State("poetry-job", "data"),
    prevent_initial_call=True,
)
def poll_poetry(_n, job_ref):
    """Drive the bar, and swap in the finished images when the job lands."""
    idle = (dash.no_update, dash.no_update, dash.no_update)

    if not job_ref or not job_ref.get("id"):
        return ("", _poetry_bar(0), True, dict(_POETRY_BOX)) + idle

    snap = jobs.MANAGER.snapshot(job_ref["id"])
    if snap is None:
        return ("", _poetry_bar(0), True, dict(_POETRY_BOX)) + idle

    frac = snap.get("fraction") or 0.0
    msg = snap.get("message") or "Working"
    detail = snap.get("detail") or ""
    line = f"{msg} — {detail}" if detail else msg

    if not jobs.MANAGER.get(job_ref["id"]).finished:
        return ((line, _poetry_bar(round(frac * 100)), False,
                 dict(_POETRY_BOX, display="flex")) + idle)

    if snap.get("error"):
        return (f"Poetry failed: {snap['error']}", _poetry_bar(0), True,
                dict(_POETRY_BOX, display="flex"),
                f"Poetry failed: {snap['error']}", dash.no_update, dash.no_update)

    job = jobs.MANAGER.get(job_ref["id"])
    result = getattr(job, "result", None) if job else None
    if not result:
        return ("", _poetry_bar(0), True, dict(_POETRY_BOX)) + idle

    items = result["items"]
    display = []
    for it in items:
        display.append(
            html.Div(
                [
                    html.H5(it["text"], style={"marginBottom": "4px", "color": "#ffc107"}),
                    html.Img(src=f"data:image/jpeg;base64,{it['poetry_img_str']}",
                             style={"width": "100%", "marginBottom": "10px"}),
                    html.Div([
                        daq.BooleanSwitch(id={"type": "select-image", "index": it["path"]},
                                          on=True, style={"display": "none"}),
                        html.Button("+ Moodboard",
                                    id={"type": "add-to-moodboard", "index": it["path"]},
                                    n_clicks=0,
                                    style={"fontSize": "12px", "padding": "2px 8px"}),
                    ], style={"display": "flex", "alignItems": "center"}),
                ],
                style={"marginBottom": "24px", "padding": "10px",
                       "backgroundColor": "#1e1e1e", "borderRadius": "5px"},
            )
        )

    updated_cache = {"story": items, "chunks": job_ref.get("chunks")}
    return ("", _poetry_bar(100), True, dict(_POETRY_BOX),
            f"Poetry-injected images saved successfully in {result['output_dir']}.",
            updated_cache, display)


@lru_cache(maxsize=40000)  # cache reads of files we've already written
def _read_cached_spec(key: str) -> bytes | None:
    f = os.path.join(SPEC_CACHE_DIR, f"{key}.png")
    try:
        with open(f, "rb") as fh:
            return fh.read()
    except Exception:
        return None

def _write_cached_spec(key: str, data: bytes) -> None:
    _paths.ensure_dir(SPEC_CACHE_DIR)
    f = os.path.join(SPEC_CACHE_DIR, f"{key}.png")
    try:
        with open(f, "wb") as fh:
            fh.write(data)
    except Exception as e:
        print("[spec-cache] write failed:", e)

def _spec_cache_key(full_path: str, mtime: float, params: tuple) -> str:
    s = f"{full_path}|{mtime}|" + "|".join(map(str, params)) + "|v3"
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def _mel_db(y: np.ndarray, sr: int, n_fft: int, hop: int, n_mels: int) -> np.ndarray:
    # Torch path is fast and can use GPU if available; falls back to librosa
    if torchaudio is not None:
        device = _gpu.device()
        wav = torch.from_numpy(y).to(device).unsqueeze(0)  # (1, T)
        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels, power=2.0
        ).to(device)(wav)  # (1, mels, frames)
        db = torchaudio.transforms.AmplitudeToDB(stype="power").to(device)(mel)
        return db.squeeze(0).detach().cpu().numpy()
    else:
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels, power=2.0)
        return librosa.power_to_db(S, ref=np.max)

def _colorize_8bit(gray: np.ndarray) -> np.ndarray:
    # gray: HxW uint8 (0..255); LUT maps to RGB
    return _SPEC_LUT[gray]  # -> HxWx3 uint8

def make_melspec_png(
    path: str,
    width_px: int = SPEC_WIDTH,
    height_px: int = SPEC_HEIGHT,
    sr: int = SPEC_PREVIEW_SR,
    seconds: int = SPEC_PREVIEW_SEC,
    n_fft: int = SPEC_NFFT,
    hop_length: int = SPEC_HOP,
    n_mels: int = SPEC_N_MELS,
) -> bytes | None:
    try:
        full = resolve_path(path)
        if not os.path.exists(full):
            return None
        mtime = os.path.getmtime(full)
        key = _spec_cache_key(full, mtime, (width_px, height_px, sr, seconds, n_fft, hop_length, n_mels))
        cached = _read_cached_spec(key)
        if cached is not None:
            return cached

        # read & downsample short window (already normalized/padded inside)
        y, _ = read_audio_mono(path, target_sr=sr, seconds=SPEC_PREVIEW_SEC, pad=False)

        # compute mel in dB (mels x frames)
        S_db = _mel_db(y, sr, n_fft, hop_length, n_mels)

        # normalize to 0..255, flip vertically to mimic origin='lower'
        S_db = np.nan_to_num(S_db, nan=np.min(S_db))
        mn, mx = float(S_db.min()), float(S_db.max())
        rng = (mx - mn) if (mx > mn) else 1.0
        img8 = ((S_db - mn) / rng * 255.0).astype(np.uint8)
        img8 = np.flipud(img8)  # put low freqs at bottom

        # resize to requested canvas
        img8 = cv2.resize(img8, (width_px, height_px), interpolation=cv2.INTER_AREA)

        # colorize via LUT and encode PNG
        rgb = _colorize_8bit(img8)  # HxWx3
        ok, buf = cv2.imencode(".png", rgb)
        if not ok:
            return None
        data = buf.tobytes()
        _write_cached_spec(key, data)
        return data
    except Exception as e:
        print("[melspec-fast] error:", e)
        return None



@app.callback(
    [
        Output("image-display", "children"),
        Output("scatter-plot", "figure"),
        Output("save-story-btn", "style"),
        Output("story-cache", "data"),
        Output("grouped-results", "data"),
        Output("carousel-state", "data"),
        Output("carousel-order", "data"),
    ],
    [
        Input("main-action-btn", "n_clicks"),
        Input("scatter-plot", "clickData"),
        Input("mode-select", "value"),
        Input("dataset-dropdown", "value"),
    ],
    [
        State("search-box", "value"),
        State("num-images", "value"),
        State("scatter-plot", "relayoutData"),
        State("story-box", "value"),
        State("group-similar", "on"),
        State("sim-thresh", "value"),
        State("spec-toggle", "on"),
    ],
)
def update_images(
    n_action, clickData, mode, dataset_value, search_value, num_images, relayoutData, story_value, group_on, sim_thresh, spec_on
):
    # Switching tabs must not throw away what you already found. mode-select is
    # an Input here only so the callback knows which branch to run when the
    # action button is pressed; on a bare mode change there is nothing to
    # recompute, so leave every output exactly as it is. Which panels are
    # visible is handled separately by toggle_inputs().
    if ctx.triggered_id == "mode-select":
        return (dash.no_update,) * 7

    # Nothing selected yet
    if not dataset_value:
        return [], _blank_fig(), {"display": "none"}, {}, [], {}, []

    # Parse "<name>::<dim>::<modality>" (backward compat to ::dim)
    try:
        parts = (dataset_value or "").split("::")
        if len(parts) == 3:
            latent_name, dim, modality = parts[0], int(parts[1]), parts[2]
        else:
            latent_name, dim, modality = parts[0], int(parts[1]), "image"
    except Exception as e:
        print(f"[update_images] bad dataset value={dataset_value!r}: {e}")
        return ([], _blank_fig("That dataset entry could not be read. Rebuild it "
                               "from the Datasets tab."),
                {"display": "none"}, {}, [], {}, [])
    db_name = latent_name


    # Load coordinates only; DO NOT load the search index yet
    df = load_data(latent_name, n_dim=dim, modality=modality)

    is_3d = all(c in df.columns for c in ["x", "y", "z"])
    color_seq = px.colors.qualitative.Dark24

    # draw the base scatter (robust to missing 'label' and 'path')
    scatter_kwargs = dict(color_discrete_sequence=color_seq)
    if "label" in df.columns:
        scatter_kwargs["color"] = "label"
    if "path" in df.columns:
        scatter_kwargs["custom_data"] = ["path"]

    if is_3d:
        fig = px.scatter_3d(df, x="x", y="y", z="z", **scatter_kwargs)
    else:
        fig = px.scatter(
            df, x="x", y="y", render_mode="webgl", **scatter_kwargs
        )

    fig.update_traces(marker=dict(size=4 if is_3d else 8))
    fig.update_layout(
        plot_bgcolor="#121212",
        paper_bgcolor="#121212",
        font=dict(color="white"),
        scene=(
            dict(
                xaxis=dict(backgroundcolor="#121212", color="white"),
                yaxis=dict(backgroundcolor="#121212", color="white"),
                zaxis=dict(backgroundcolor="#121212", color="white"),
            )
            if is_3d
            else {}
        ),
    )
    
    if is_3d:
        fig.update_traces(marker=dict(opacity=0.6), selector=dict(type="scatter3d"))
    else:
        fig.update_traces(marker=dict(opacity=0.6), selector=dict(type="scattergl"))

    # keep camera/zoom across updates
    fig.update_layout(uirevision="keep")

    trigger = ctx.triggered_id if hasattr(ctx, "triggered_id") else None
    print(f"[update_images] trigger={trigger} mode={mode} dataset={dataset_value}")

    images = []
    show_save_story = {"display": "none"}
    story_cache = {}
    groups_store = []  # must be a list (not {})
    car_state_store = {}

    # --- STORY mode (unchanged logic, but return empty group/carousel stores) ---
    if mode == "story" and trigger == "main-action-btn" and story_value:
        index, idx2path = load_index(db_name, modality=modality)
        print("[DEBUG] STORY mode triggered")
        story_chunks = [chunk.strip() for chunk in story_value.split("\n") if chunk.strip()]
        print(f"[DEBUG] Story chunks: {story_chunks}")
        story_images = []
        for i, chunk in enumerate(story_chunks):
            results = search(index, idx2path, chunk, 1, modality=modality)
            print(f"[DEBUG] Search results for chunk '{chunk}': {results}")
            if results:
                _, path, _ = results[0]
                story_images.append({"text": chunk, "path": path, "img_str": ""})

        if story_images:
            coords, story_texts = [], []
            for s in story_images:
                row = df[df["path"] == s["path"]]
                story_texts.append(s["text"])
                if is_3d:
                    coords.append((row["x"].values[0], row["y"].values[0], row["z"].values[0]))
                else:
                    coords.append((row["x"].values[0], row["y"].values[0]))
            if is_3d:
                xs, ys, zs = zip(*coords)
                fig.add_trace(
                    go.Scatter3d(
                        x=xs,
                        y=ys,
                        z=zs,
                        mode="lines+markers",
                        line=dict(color="gold", width=4),
                        marker=dict(size=8, symbol="cross"),
                        text=story_texts,
                        hovertemplate="%{text}<extra></extra>",
                        name="Story Path",
                    )
                )
            else:
                xs, ys = zip(*coords)
                fig.data = tuple(t for t in fig.data if getattr(t, "name", None) != "Story Path")
                # Scattergl for the same reason as Search Results: an SVG line
                # would be drawn underneath the WebGL point cloud.
                fig.add_trace(
                    go.Scattergl(
                        x=xs,
                        y=ys,
                        mode="lines+markers",
                        line=dict(color="gold", width=4),
                        marker=dict(size=14, color="gold",
                                    line=dict(width=1, color="#ffffff")),
                        text=story_texts,
                        hovertemplate="%{text}<extra></extra>",
                        name="Story Path",
                        legendgroup="storypath",
                        showlegend=True,
                    )
                )
                fig.add_annotation(
                    x=xs[0],
                    y=ys[0],
                    text="Beginning",
                    showarrow=False,
                    font=dict(size=14, color="gold"),
                    yshift=28,
                    bgcolor="rgba(0,0,0,0.5)",
                    borderpad=6,
                )

        for img in story_images:
            qpath = urllib.parse.quote(img["path"])
            if modality == "image":
                media = html.Img(
                    src=f"/preview?p={qpath}&w=900",
                    srcSet=f"/preview?p={qpath}&w=600 600w, /preview?p={qpath}&w=900 900w, /preview?p={qpath}&w=1400 1400w",
                    sizes="(max-width: 900px) 90vw, 42vw",
                    style={"width": "100%", "marginBottom": "10px"},
                )
            else:
                preview_endpoint = "/aspec" if spec_on else "/awave"
                media = html.Div([
                    html.Img(src=f"{preview_endpoint}?p={qpath}", style={"width": "100%", "marginBottom": "8px"}),
                    html.Audio(src=f"/audio?p={qpath}", controls=True, style={"width": "100%"}),
                ])
            # Story results are ordinary images from the same dataset, so they
            # get the same "+ Moodboard" action the prompt-search cards have.
            # Without it a picture found by writing a scene could only be
            # collected by going back to Prompt Search and hunting for it
            # again. The pattern id matches the one update_moodboard already
            # listens to with ALL, so no new callback is needed.
            images.append(
                html.Div(
                    [
                        html.H5(img["text"], style={"marginBottom": "4px", "color": "#ffc107"}),
                        media,
                        html.Div(
                            html.Button(
                                "+ Moodboard",
                                id={"type": "add-to-moodboard", "index": img["path"]},
                                n_clicks=0,
                                style={"fontSize": "12px", "padding": "2px 8px"},
                            ),
                            style={"display": "flex", "alignItems": "center"},
                        ),
                    ],
                    style={"marginBottom": "24px", "padding": "10px", "backgroundColor": "#1e1e1e", "borderRadius": "5px"},
                )
            )

        show_save_story = {"display": "block", "marginTop": "10px"}
        story_cache = {"story": story_images, "chunks": story_chunks}
        return images, fig, show_save_story, story_cache, groups_store, car_state_store, []

    # --- PROMPT mode with grouping / carousel ---
    if mode == "prompt" and trigger == "main-action-btn" and search_value:
        print("[DEBUG] PROMPT mode triggered")
        index, idx2path = load_index(db_name, modality=modality)
        results = search(index, idx2path, search_value, num_images, modality=modality)

        print(f"[DEBUG] Search results: {results}")
        if len(results):
            # A usearch key is NOT a DataFrame row label. The latent frame is built
            # by walking idx2path in insertion order and is then reset to a 0..N-1
            # RangeIndex, so key -> row must go through that same ordering. They
            # only coincide when no file failed to read during indexing; when one
            # did, .loc silently highlighted the wrong points or raised KeyError.
            key_to_row = {k: i for i, k in enumerate(idx2path.keys())}
            rows = [key_to_row[r[0]] for r in results if r[0] in key_to_row]
            dropped = len(results) - len(rows)
            if dropped:
                print(f"[WARN] {dropped} search result(s) have no row in the latent space; "
                      f"the index and latent file are out of sync for '{db_name}'.")
            highlighted_df = df.iloc[rows]
            print(f"[DEBUG] Highlighted DataFrame: {highlighted_df.shape[0]} rows")

            if is_3d:
                fig.add_trace(
                    go.Scatter3d(
                        x=highlighted_df["x"], y=highlighted_df["y"], z=highlighted_df["z"],
                        mode="markers",
                        marker=dict(size=10, symbol="cross", opacity=1),
                        name="Search Results",
                    )
                )
            else:
                # remove any previous "Search Results" trace so you don't stack them
                fig.data = tuple(t for t in fig.data if getattr(t, "name", None) != "Search Results")

                xs = highlighted_df["x"].to_list()
                ys = highlighted_df["y"].to_list()

                # The base scatter is WebGL. Plotly draws the GL canvas ABOVE the
                # SVG trace layer, so an SVG go.Scatter overlay ends up hidden
                # behind the dots. Use Scattergl so the highlight lives in the
                # same GL layer, where trace order decides what is on top.
                fig.add_trace(
                    go.Scattergl(
                        x=xs, y=ys,
                        mode="markers",
                        marker=dict(
                            symbol="x",
                            size=20,
                            color="#33C3F0",
                            line=dict(width=2, color="#ffffff"),
                        ),
                        name="Search Results",
                        hoverinfo="skip",
                        showlegend=True,
                        opacity=1.0,
                    )
                )




        keys = [k for (k, p, d) in results]
        paths = [p for (k, p, d) in results]
        print(f"[DEBUG] Grouping keys: {keys}")
        print(f"[DEBUG] Grouping paths: {paths}")
        if group_on:
            groups = _cosine_group(keys, paths, index, float(sim_thresh or 0.08))
            print(f"[DEBUG] Grouped results: {groups}")
        else:
            groups = [{"gid": f"g{i}", "keys": [keys[i]], "paths": [paths[i]]} for i in range(len(keys))]
            print(f"[DEBUG] Ungrouped results: {groups}")

        car_state = {g["gid"]: 0 for g in groups}
        carousel_order = [g["gid"] for g in groups if len(g.get("paths", [])) > 1]
        print(f"[DEBUG] Carousel state: {car_state}")
        cards = []

        for g in groups:
            n = len(g["paths"])
            first = g["paths"][0]
            qpath = urllib.parse.quote(first)

            if n == 1:
                if modality == "image":
                    preview = html.Img(
                        src=f"/preview?p={qpath}&w=900",
                        srcSet=f"/preview?p={qpath}&w=600 600w, /preview?p={qpath}&w=900 900w, /preview?p={qpath}&w=1400 1400w",
                        sizes="(max-width: 900px) 90vw, 42vw",
                        style={"width": "100%", "marginBottom": "10px"},
                    )
                else:
                    preview = html.Div([
                        html.Img(src=f"{('/aspec' if spec_on else '/awave')}?p={qpath}", style={"width": "100%", "marginBottom": "6px"}),
                        html.Audio(src=f"/audio?p={qpath}", controls=True, style={"width": "100%"}),
                    ])

                cards.append(
                    html.Div(
                        [
                            preview,
                            html.Div([
                                daq.BooleanSwitch(id={"type": "select-image", "index": first}, on=False),
                                html.Button("+ Moodboard", id={"type": "add-to-moodboard", "index": first}, 
                                           n_clicks=0, style={"marginLeft": "10px", "fontSize": "12px", "padding": "2px 8px"}),
                            ], style={"display": "flex", "alignItems": "center"}),
                            html.Span(" (no twins)", style={"marginLeft": "10px", "opacity": 0.7}),
                        ],
                        style={"marginBottom": "20px", "padding": "10px", "backgroundColor": "#1e1e1e", "borderRadius": "5px"},
                    )
                )
            else:
                # carousel
                if modality == "image":
                    media_el = html.Img(
                        id={"type": "carousel-img", "gid": g["gid"]},
                        src=f"/preview?p={qpath}&w=900",
                        srcSet=f"/preview?p={qpath}&w=600 600w, /preview?p={qpath}&w=900 900w, /preview?p={qpath}&w=1400 1400w",
                        sizes="(max-width: 900px) 90vw, 42vw",
                        style={"width": "100%", "display": "block", "marginBottom": "10px", "borderRadius": "5px"},
                    )
                    extra_player = []
                else:
                    media_el = html.Img(
                        id={"type": "carousel-img", "gid": g["gid"]},
                        src=f"{('/aspec' if spec_on else '/awave')}?p={qpath}",
                        style={"width": "100%", "display": "block", "marginBottom": "6px", "borderRadius": "5px"},
                    )

                    extra_player = [html.Audio(id={"type": "carousel-audio", "gid": g["gid"]},
                                            src=f"/audio?p={qpath}", controls=True, style={"width": "100%"})]

                cards.append(
                    html.Div(
                        [
                            html.Div(
                                [
                                    media_el,
                                    html.Button("◀", id={"type": "left", "gid": g["gid"]}, n_clicks=0,
                                                style={"position": "absolute", "left": "8px", "top": "50%",
                                                    "transform": "translateY(-50%)",
                                                    "backgroundColor": "rgba(0,0,0,0.6)", "color": "#fff",
                                                    "border": "none", "borderRadius": "9999px",
                                                    "width": "36px", "height": "36px", "zIndex": 2, "cursor": "pointer"}),
                                    html.Button("▶", id={"type": "right", "gid": g["gid"]}, n_clicks=0,
                                                style={"position": "absolute", "right": "8px", "top": "50%",
                                                    "transform": "translateY(-50%)",
                                                    "backgroundColor": "rgba(0,0,0,0.6)", "color": "#fff",
                                                    "border": "none", "borderRadius": "9999px",
                                                    "width": "36px", "height": "36px", "zIndex": 2, "cursor": "pointer"}),
                                ],
                                style={"position": "relative", "overflow": "hidden"},
                            ),
                            *extra_player,
                            html.Div(id={"type": "carousel-counter", "gid": g["gid"]}, children=f"1/{n}",
                                    style={"textAlign": "center", "margin": "4px 0 8px 0", "opacity": 0.8}),
                            html.Div([
                                daq.BooleanSwitch(id={"type": "select-image", "index": f"group::{g['gid']}"}, on=False),
                                html.Button("+ Moodboard", id={"type": "add-to-moodboard", "index": first}, 
                                           n_clicks=0, style={"marginLeft": "10px", "fontSize": "12px", "padding": "2px 8px"}),
                            ], style={"display": "flex", "alignItems": "center"}),
                            html.Span(f" twins: {n}", style={"marginLeft": "10px", "opacity": 0.7}),
                        ],
                        style={"marginBottom": "20px", "padding": "10px", "backgroundColor": "#1e1e1e",
                            "borderRadius": "5px", "overflowX": "hidden"},
                    )
                )

        print(f"[DEBUG] Returning {len(cards)} cards")
        # after you compute `groups` and before returning cards:
        try:
            if modality == "audio" and spec_on:
                cand = []
                for g in groups:
                    cand.extend(g.get("paths", []))
                # precompute the first N most likely to be viewed
                for p in cand[:64]:
                    SPEC_EXEC.submit(make_melspec_png, p)
        except Exception as e:
            print("[spec prewarm] skipped:", e)

        return cards, fig, {"display": "none"}, {}, groups, car_state, carousel_order

    # --- Scatter click (unchanged) ---
    if trigger == "scatter-plot" and clickData:
        pt = clickData["points"][0]
        custom = pt.get("customdata") or []
        media_path = custom[0] if custom else None
        if not media_path:
            return images, fig, {"display": "none"}, {}, [], {}, []

        qpath = urllib.parse.quote(media_path)
        ext = os.path.splitext(media_path)[1].lower()
        if ext in AUDIO_EXTS:
            media = html.Div([
                html.Img(src=f"/awave?p={qpath}", style={"width": "100%", "marginBottom": "6px"}),
                html.Audio(src=f"/audio?p={qpath}", controls=True, style={"width": "100%"}),
            ])
        else:
            media = html.Img(
                src=f"/preview?p={qpath}&w=900",
                srcSet=f"/preview?p={qpath}&w=600 600w, /preview?p={qpath}&w=900 900w, /preview?p={qpath}&w=1400 1400w",
                sizes="(max-width: 900px) 90vw, 42vw",
                style={"width": "100%", "marginBottom": "10px"},
            )

        images.append(
            html.Div(
                [
                    media,
                    html.Div([
                        daq.BooleanSwitch(id={"type": "select-image", "index": media_path}, on=False),
                        html.Button("+ Moodboard", id={"type": "add-to-moodboard", "index": media_path}, 
                                   n_clicks=0, style={"marginLeft": "10px", "fontSize": "12px", "padding": "2px 8px"}),
                    ], style={"display": "flex", "alignItems": "center"}),
                ],
                style={"marginBottom": "20px", "padding": "10px", "backgroundColor": "#1e1e1e", "borderRadius": "5px"},
            )
        )
        return images, fig, {"display": "none"}, {}, [], {}, []


    # keep camera on pan/zoom
    if is_3d and relayoutData and "scene.camera" in relayoutData:
        fig.update_layout(scene_camera=relayoutData["scene.camera"])

    return images, fig, show_save_story, story_cache, groups_store, car_state_store, []


@app.callback(
    [
        Output("carousel-state", "data", allow_duplicate=True),
        Output({"type": "carousel-img", "gid": ALL}, "src"),
        Output({"type": "carousel-img", "gid": ALL}, "srcSet"),
        Output({"type": "carousel-counter", "gid": ALL}, "children"),
        Output({"type": "carousel-audio", "gid": ALL}, "src"),
    ],
    [
        Input({"type": "left", "gid": ALL}, "n_clicks"),
        Input({"type": "right", "gid": ALL}, "n_clicks"),
        Input("grouped-results", "data"),
        Input("carousel-order", "data"),
    ],
    [
        State("carousel-state", "data"),
        State("spec-toggle", "on"),
    ],
    prevent_initial_call=True,
)
def nav_carousel(left_clicks, right_clicks, groups, order, car_state, spec_on):
    """
    Robust carousel navigation:

    - Uses the number of actual carousel components (len(left_clicks)) as
      the ground truth for how many values to return.
    - Derives audio vs image behaviour from the stored groups.
    - Keeps car_state consistent with currently existing groups only.
    """

    groups = groups or []
    order = order or []
    car_state = dict(car_state or {})

    # Only groups that actually have "twins"
    car_groups = {g["gid"]: g for g in groups if len(g.get("paths", [])) > 1}

    # How many carousel components are currently mounted in the layout?
    n_components = len(left_clicks or [])  # equals number of left/right buttons & imgs
    if not car_groups or n_components == 0:
        # No carousels to drive → return empty lists of the correct size
        return car_state, [], [], [], []

    # Determine in which order the carousels appear in the layout
    # Prefer the stored "order" (built when cards are created), but
    # intersect it with existing groups just in case.
    if order:
        gid_list = [gid for gid in order if gid in car_groups]
    else:
        # fallback: use the dict order
        gid_list = list(car_groups.keys())

    # Clamp to the current number of components (safety against stale order)
    gid_list = gid_list[:n_components]

    # Ensure state only contains current groups
    for gid in gid_list:
        car_state.setdefault(gid, 0)
    car_state = {gid: car_state[gid] for gid in gid_list}

    # Which dataset type? (image vs audio) → determines if we have audio carousels
    sample_gid = gid_list[0]
    sample_path = car_groups[sample_gid]["paths"][0]
    is_audio_dataset = os.path.splitext(sample_path)[1].lower() in AUDIO_EXTS

    # Handle user click (if any)
    trig = ctx.triggered_id
    if isinstance(trig, dict) and trig.get("type") in ("left", "right"):
        gid = trig.get("gid")
        if gid in car_groups:
            paths = car_groups[gid]["paths"]
            n = len(paths)
            cur = car_state.get(gid, 0)
            if trig["type"] == "left":
                cur = (cur - 1) % n
            else:
                cur = (cur + 1) % n
            car_state[gid] = cur

    srcs = []
    srcsets = []
    counters = []
    audios = []

    for gid in gid_list:
        g = car_groups[gid]
        paths = g["paths"]
        cur = car_state.get(gid, 0) % len(paths)
        qp = urllib.parse.quote(paths[cur])

        if is_audio_dataset:
            # Image is a spectrogram or waveform, plus an <audio> element
            preview_ep = "/aspec" if spec_on else "/awave"
            srcs.append(f"{preview_ep}?p={qp}")
            srcsets.append(dash.no_update)  # no srcset needed for spectrogram
            audios.append(f"/audio?p={qp}")
        else:
            # Normal image carousel
            srcs.append(f"/preview?p={qp}&w=900")
            srcsets.append(
                f"/preview?p={qp}&w=600 600w, "
                f"/preview?p={qp}&w=900 900w, "
                f"/preview?p={qp}&w=1400 1400w"
            )

        counters.append(f"{cur + 1}/{len(paths)}")

    # If the layout has somehow more components than gids (extremely defensive),
    # pad with dash.no_update so Dash's lengths always match.
    def _pad(lst, target_len):
        if len(lst) < target_len:
            lst = list(lst) + [dash.no_update] * (target_len - len(lst))
        else:
            lst = lst[:target_len]
        return lst

    srcs = _pad(srcs, n_components)
    srcsets = _pad(srcsets, n_components)
    counters = _pad(counters, n_components)

    if is_audio_dataset:
        audios = _pad(audios, n_components)
    else:
        # For image datasets, there are *no* carousel-audio components,
        # so the output list must be empty.
        audios = []

    return car_state, srcs, srcsets, counters, audios


@app.callback(
    Output("hover-thumb", "src"),
    Output("hover-thumb", "style"),
    Input("scatter-plot", "hoverData"),
    Input("mode-select", "value"),
    State("dataset-dropdown", "value"),
    State("spec-toggle", "on"),
)
def update_hover_thumb(hoverData, mode, dataset_value, spec_on):
    # The hover preview belongs to the scatter, so it must go wherever the
    # scatter goes. Listing the modes that DO show a scatter, rather than the
    # ones that don't, means adding a mode cannot strand a thumbnail on screen
    # again -- which is exactly what happened when "datasets" was added.
    if mode not in ("prompt", "story"):
        return "", HOVER_THUMB_HIDDEN
    
    try:
        parts = (dataset_value or "").split("::")
        modality = parts[2] if len(parts) == 3 else "image"
        if hoverData and "points" in hoverData:
            pt = hoverData["points"][0]
            custom = pt.get("customdata") or []
            if custom:
                media_path = custom[0]
                if modality == "audio":
                    endpoint = "/aspec?p=" if spec_on else "/awave?p="
                else:
                    endpoint = "/thumb?p="
                thumb_url = endpoint + urllib.parse.quote(media_path)
                return thumb_url, HOVER_THUMB_SHOWN
    except Exception as e:
        print("[hover-thumb] skipped due to:", e)
    return dash.no_update, HOVER_THUMB_HIDDEN




@app.callback(
    [
        Output("relocate-panel", "style"),
        Output("relocate-message", "children"),
        Output("relocate-root", "value"),
        Output("relocate-status", "children", allow_duplicate=True),
    ],
    Input("dataset-dropdown", "value"),
    prevent_initial_call="initial_duplicate",
)
def detect_moved_dataset(dataset_value):
    """
    Notice when a dataset's media is no longer where its index says.

    Samples a couple of hundred paths rather than stat-ing all 82k, because this
    runs on every dataset change.
    """
    hidden = {"display": "none"}
    if not dataset_value:
        return hidden, "", "", ""
    try:
        name, _dim, modality = _parse_dataset_value(dataset_value)
    except Exception:
        return hidden, "", "", ""

    try:
        from .relocate import dataset_health
    except ImportError:
        from relocate import dataset_health
    try:
        h = dataset_health(name, modality)
    except Exception as e:
        return hidden, "", "", f"Could not check this dataset: {e}"

    if h["ok"] and not h["error"]:
        return hidden, "", "", ""

    shown = {"display": "block", "marginBottom": "18px", "padding": "12px 14px",
             "backgroundColor": "#2a2318", "border": "1px solid #5a4a22",
             "borderRadius": "8px"}
    if h["error"]:
        return shown, f"⚠ {h['error']}", "", ""

    frac = f"{h['missing']} of {h['checked']} sampled"
    if h["checked"] < h["total"]:
        frac += f" (dataset has {h['total']:,} items)"
    msg = html.Div([
        html.Div(f"⚠ This dataset's files are missing — {frac}.",
                 style={"fontWeight": "600", "marginBottom": "3px"}),
        html.Div(["Indexed from ", html.Code(h["root"] or "an unknown folder",
                                             style={"color": "#d0c0a0"}),
                  ". If that folder moved or lives on a drive that is not connected, "
                  "point Arcana at its new location below — files are matched by "
                  "content, so nothing is re-indexed."],
                 style={"color": "#b8a88a", "fontSize": "12px", "lineHeight": "1.5"}),
    ])
    return shown, msg, (h["root"] or ""), ""


@app.callback(
    Output("relocate-status", "children", allow_duplicate=True),
    [Input("relocate-check-btn", "n_clicks"), Input("relocate-apply-btn", "n_clicks")],
    [State("relocate-root", "value"), State("dataset-dropdown", "value")],
    prevent_initial_call=True,
)
def do_relocate(_check, _apply, new_root, dataset_value):
    """Dry-run or apply a relocation for the selected dataset."""
    if not dataset_value:
        return "Select a dataset first."
    if not new_root or not str(new_root).strip():
        return html.Span("Enter the folder the files live in now.", style={"color": "#e0a44a"})
    new_root = os.path.expanduser(str(new_root).strip().strip('"'))
    if not os.path.isdir(new_root):
        return html.Span(f"Not a folder: {new_root}", style={"color": "#e74c3c"})

    apply_it = ctx.triggered_id == "relocate-apply-btn"
    try:
        name, _dim, modality = _parse_dataset_value(dataset_value)
        from .relocate import relocate_legacy
        from .legacy import discover
    except ImportError:
        from relocate import relocate_legacy
        from legacy import discover
    except Exception as e:
        return html.Span(f"Could not parse the dataset: {e}", style={"color": "#e74c3c"})

    matches = [d for d in discover() if d.name == name and d.modality == modality]
    if not matches:
        return html.Span(f"No dataset named {name}.", style={"color": "#e74c3c"})

    try:
        r = relocate_legacy(matches[0], new_root, dry_run=not apply_it)
    except Exception as e:
        return html.Span(f"Relocation failed: {type(e).__name__}: {e}",
                         style={"color": "#e74c3c"})

    if r["found"] == 0:
        return html.Span(
            f"Found none of the {r['total']:,} files under {new_root}. "
            f"Point at the folder that directly contains them (or their subfolders).",
            style={"color": "#e74c3c"})

    if not apply_it:
        extra = f" — {r['missing']} would still be missing." if r["missing"] else ""
        return html.Span(
            f"Would match {r['found']:,} of {r['total']:,} files here{extra} "
            f"Press Relocate to apply.", style={"color": "#4caf50"})

    # Applied: stale absolute paths are cached all over the place.
    make_thumbnail_bytes.cache_clear()
    make_resized_bytes.cache_clear()
    thumb_b64_for.cache_clear()
    with _ALLOWED_ROOTS_LOCK:
        _REGISTERED_DATASETS.clear()
    register_media_root(new_root)

    tail = f" {r['missing']} still missing." if r["missing"] else ""
    return html.Span(
        f"✓ Relocated {r['found']:,} of {r['total']:,} files.{tail} "
        f"Re-select the dataset to reload it. Backups were written next to the "
        f"originals as .bak.", style={"color": "#4caf50"})


@app.callback(
    Output("results-owner", "data"),
    [
        Input("main-action-btn", "n_clicks"),
        Input("moodboard-search-btn", "n_clicks"),
        Input("scatter-plot", "clickData"),
    ],
    State("mode-select", "value"),
    prevent_initial_call=True,
)
def track_results_owner(_action, _moodboard, _click, mode):
    """
    Record which mode last filled the shared results panel.

    image-display is written by both update_images() and the moodboard
    similarity search, so without this the panel shows one mode's results while
    you are looking at another.
    """
    if ctx.triggered_id == "moodboard-search-btn":
        return "moodboard"
    return mode


@app.callback(
    Output("image-display", "style"),
    [Input("mode-select", "value"), Input("results-owner", "data")],
)
def toggle_results_visibility(mode, owner):
    """
    Keep results across tab switches, but only show them in the tab that made
    them. Switching away hides the panel without discarding it, so coming back
    restores what was there.
    """
    # The right column already scrolls, so this must not add a second scrollbar
    # or an 80vh box taller than the space it sits in.
    base = {"overflowX": "hidden"}
    if owner is None or owner == mode:
        return {**base, "display": "block"}
    return {**base, "display": "none"}


@app.callback(Output("save-button", "style"), Input("mode-select", "value"))
def toggle_save_selected_button(mode):
    if mode == "prompt":
        return {"marginTop": "10px", "display": "block"}
    else:
        return {"display": "none"}


@app.callback(
    Output("poetry-inline", "style"),
    Input("mode-select", "value"),
    Input("dataset-dropdown", "value"),
)
def toggle_poetry_inline(mode, dataset_value):
    parts = (dataset_value or "").split("::")
    modality = parts[2] if len(parts) == 3 else "image"
    show = (mode == "story") and (modality == "image")
    base = {"alignItems": "center", "gap": "10px"}
    return {**base, "display": "flex"} if show else {**base, "display": "none"}


@app.callback(
    Output("audio-spec-inline", "style"),
    Input("dataset-dropdown", "value"),
)
def toggle_spec_inline(dataset_value):
    parts = (dataset_value or "").split("::")
    modality = parts[2] if len(parts) == 3 else "image"
    base = {"alignItems": "center", "gap": "10px", "marginLeft": "12px"}
    return {**base, "display": "flex"} if modality == "audio" else {**base, "display": "none"}


@app.callback(
    Output("save-confirmation", "children"),
    [Input("save-button", "n_clicks"), Input("save-story-btn", "n_clicks")],
    [
        State({"type": "select-image", "index": dash.ALL}, "on"),
        State({"type": "select-image", "index": dash.ALL}, "id"),
        State("save-folder", "value"),
        State("mode-select", "value"),
        State("story-cache", "data"),
        State("grouped-results", "data"),
        State("carousel-state", "data"),
    ],
)
def save_images(n_clicks_images, n_clicks_story, selections, ids, folder, mode, story_cache, groups, car_state):
    msg = ""
    triggered = ctx.triggered_id if hasattr(ctx, "triggered_id") else None

    # helper: which path is currently active for a group gid
    def _current_path_for_gid(gid: str):
        g = next((x for x in (groups or []) if x["gid"] == gid), None)
        if not g:
            return None
        cur = (car_state or {}).get(gid, 0) % max(1, len(g["paths"]))
        return g["paths"][cur]

    if triggered == "save-button":
        subfolder = folder or "session"
        save_dir = _safe_output_dir("selections", subfolder)
        os.makedirs(save_dir, exist_ok=True)

        selections = selections or []
        ids = ids or []
        selected_paths = []
        for id_obj, selected in zip(ids, selections):
            if not selected:
                continue
            idx = id_obj.get("index")
            if isinstance(idx, str) and idx.startswith("group::"):
                gid = idx.split("::", 1)[1]
                p = _current_path_for_gid(gid)
                if p:
                    selected_paths.append(p)
            else:
                selected_paths.append(idx)

        n_saved = 0
        for path in selected_paths:
            if not path:
                continue
            full_path = resolve_path(path)
            basename = os.path.basename(full_path)
            prefix = hashlib.md5(path.encode("utf-8")).hexdigest()[:8]
            safe_name = f"{prefix}_{basename}"

            ext = os.path.splitext(full_path)[1].lower()
            if ext in AUDIO_EXTS:
                # copy audio as-is
                try:
                    shutil.copy2(full_path, os.path.join(save_dir, safe_name))
                    n_saved += 1
                except Exception as e:
                    print(f"[ERROR] Could not copy audio: {full_path} ({e})")
            else:
                # try image write
                img = imread_unicode(full_path)
                if img is not None:
                    imwrite_unicode(os.path.join(save_dir, safe_name), img)
                    n_saved += 1
                else:
                    # fallback: copy raw file if not an image
                    try:
                        shutil.copy2(full_path, os.path.join(save_dir, safe_name))
                        n_saved += 1
                    except Exception as e:
                        print(f"[ERROR] Could not save: {full_path} ({e})")
        msg = f"{n_saved} files saved successfully to {save_dir}."


    elif triggered == "save-story-btn":
        subfolder = folder or "story"
        save_dir = _safe_output_dir("stories", subfolder)
        poetry_dir = os.path.join(save_dir, "poetry_injected")
        original_dir = os.path.join(save_dir, "original")
        os.makedirs(poetry_dir, exist_ok=True)
        os.makedirs(original_dir, exist_ok=True)

        n_saved = 0
        if story_cache and "story" in story_cache:
            for i, item in enumerate(story_cache["story"]):
                full_img_path = resolve_path(item["path"])
                original_img = imread_unicode(full_img_path)
                if original_img is not None:
                    imwrite_unicode(os.path.join(original_dir, f"{i:02d}_original.jpg"), original_img)
                    n_saved += 1
                poetry_img_path = item.get("poetry_img_path")
                if poetry_img_path and os.path.exists(poetry_img_path):
                    poetry_img = imread_unicode(poetry_img_path)
                    if poetry_img is not None:
                        imwrite_unicode(os.path.join(poetry_dir, f"{i:02d}_poetry.jpg"), poetry_img)
                        n_saved += 1

            with open(os.path.join(save_dir, "story.txt"), "w", encoding="utf-8") as f:
                for i, chunk in enumerate(story_cache.get("chunks", [])):
                    f.write(f"{i+1}. {chunk}\n")

            msg = f"Story and {n_saved} images (original + poetry-injected) saved successfully to {save_dir}."
        else:
            msg = "No story to save."
    return msg


# The dataset manager registers its own callbacks; done here so the layout
# it refers to already exists.
_ui_datasets.register(app)


def main():
    # 8051, not Dash's default 8050. The packaged app (installer/launcher.py)
    # listens on 8050, so a development server on the same port either refuses
    # to start or, worse, quietly serves the installed app's URL from a
    # different codebase -- and because the moodboard collection lives in
    # browser localStorage, which is keyed by origin, the two would also share
    # state and look like one confusing application.
    #
    # Override with ARCANA_DEV_PORT when running several branches side by side.
    port = int(os.environ.get("ARCANA_DEV_PORT", "8051"))
    app.run(host="127.0.0.1", port=port, debug=False)


if __name__ == "__main__":
    main()

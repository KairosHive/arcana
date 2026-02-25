import dash
from dash import dcc, html, Input, Output, State, ctx
import plotly.express as px
import pandas as pd
import cv2
import base64
import os
import pickle
import torch
import threading
import json
from usearch.index import Index
from transformers import CLIPModel, CLIPProcessor
from concurrent.futures import ThreadPoolExecutor
import dash_daq as daq
import numpy as np
import plotly.graph_objects as go
import re
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
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





# --- audio + CLAP ---
import soundfile as sf
try:
    import torchaudio
except Exception:
    torchaudio = None
from transformers import ClapModel, ClapProcessor



torch.set_grad_enabled(False)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
LATENTS_DIR = os.path.join(APP_ROOT, "latents")
DB_DIR = os.path.join(APP_ROOT, "databases")

IMAGES_ROOT = os.path.abspath(os.path.join(APP_ROOT, "..", "images"))

OUTPUT_DIR = os.path.join(APP_ROOT, "output")
STORIES_DIR = os.path.join(OUTPUT_DIR, "stories")
SELECTIONS_DIR = os.path.join(OUTPUT_DIR, "selections")
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}

# ---- FAST SPEC CONFIG + CACHE ----
SPEC_CACHE_DIR = os.path.join(APP_ROOT, "cache_specs")
os.makedirs(SPEC_CACHE_DIR, exist_ok=True)

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



os.makedirs(STORIES_DIR, exist_ok=True)
os.makedirs(SELECTIONS_DIR, exist_ok=True)


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


@lru_cache(maxsize=50000)
def make_thumbnail_bytes(path: str, max_side: int = 192) -> bytes | None:
    full_path = resolve_path(path)
    img = cv2.imread(full_path)
    if img is None:
        return None
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


@lru_cache(maxsize=2000)  # cache a few thousand medium previews
def make_resized_bytes(path: str, max_w: int = 900, q: int = 72) -> bytes | None:
    full_path = resolve_path(path)
    img = cv2.imread(full_path)
    if img is None:
        return None
    h, w = img.shape[:2]
    if w > max_w:
        new_w = max_w
        new_h = int(h * (max_w / float(w)))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    return buf.tobytes() if ok else None


@lru_cache(maxsize=50000)  # 13k fits
def thumb_b64_for(path: str) -> str | None:
    return encode_thumbnail(path)  # uses resolve_path inside


# --- replace current signature + body ---
def read_audio_mono(path, target_sr=24000, seconds=None, pad=False):
    full = resolve_path(path)
    if torchaudio is not None:
        wav, sr = torchaudio.load(full)  # (ch, n)
        wav = wav.mean(dim=0).numpy()
    else:
        wav, sr = sf.read(full, always_2d=False)
        if wav.ndim == 2:
            wav = wav.mean(axis=1)

    # resample
    if sr != target_sr:
        if torchaudio is not None:
            wav = torchaudio.functional.resample(torch.from_numpy(wav), sr, target_sr).numpy()
        else:
            import librosa
            wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)

    # crop (no right-pad unless pad=True)
    if seconds is not None:
        max_len = int(target_sr * seconds)
        if wav.shape[0] > max_len:
            wav = wav[:max_len]
        elif pad:
            wav = np.pad(wav, (0, max_len - wav.shape[0]))
    return wav.astype(np.float32), target_sr


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
        return None


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
    for fname in os.listdir(latent_dir):
        m = pattern.match(fname)
        if m:
            options.append({"label": m.group(1), "value": m.group(1)})
    return sorted(options, key=lambda x: x["label"])


def resolve_path(p):
    # If the stored path is absolute, use it; else fall back to IMAGES_ROOT
    return p if os.path.isabs(p) else os.path.join(IMAGES_ROOT, p)


def get_db_options(db_dir=DB_DIR):
    pattern = re.compile(r"index_(.+)\.pkl$")
    options = []
    for fname in os.listdir(db_dir):
        m = pattern.match(fname)
        if m:
            options.append({"label": m.group(1), "value": m.group(1)})
    return sorted(options, key=lambda x: x["label"])


def get_matching_datasets(latent_dir=LATENTS_DIR, db_dir=DB_DIR):
    # Files: latent_space_<name>_<mod>_<dim>d.pkl  &  index_<name>_<mod>.pkl
    lat_pat = re.compile(r"latent_space_(.+)_(image|audio)_(\d+)d\.pkl$")
    db_pat  = re.compile(r"index_(.+)_(image|audio)\.pkl$")
    lat_map = {}
    for fname in os.listdir(latent_dir):
        m = lat_pat.match(fname)
        if m:
            name, mod, dim = m.group(1), m.group(2), m.group(3)
            lat_map.setdefault((name, mod), []).append(dim)
    db_keys = {(m.group(1), m.group(2)) for fname in os.listdir(db_dir) if (m := db_pat.match(fname))}
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
    image = cv2.imread(full_path)
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
    img = cv2.imread(full_path)
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
    for col in ("x", "y", "z"):
        if col in df.columns:
            df[col] = df[col].astype("float32")
    return df.reset_index(drop=True)

def load_index(name, modality="image"):
    index_name = os.path.join(DB_DIR, f"index_{name}_{modality}.pkl")
    with open(index_name, "rb") as f:
        idx_blob, idx2path = pickle.load(f)
    return Index.restore(idx_blob), idx2path



# ------------- CLIP MODEL LOAD ONCE -------------
model = CLIPModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K", torch_dtype=torch.float16)
# model.eval().to("cuda")
model.eval().to("cpu")
processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")

# --- lazy CLAP (keep FP32 to avoid BN/dtype issues) ---
_CLAP = {"model": None, "proc": None}
def load_clap(device="cpu"):
    if _CLAP["model"] is None:
        m = ClapModel.from_pretrained("laion/clap-htsat-fused")
        m.eval().to(device)
        p = ClapProcessor.from_pretrained("laion/clap-htsat-fused")
        _CLAP.update(model=m, proc=p)
    return _CLAP["model"], _CLAP["proc"]


def search(index, idx2path, query, n, modality="image"):
    if modality == "image":
        inputs = processor.tokenizer(query, return_tensors="pt")
        vec = model.get_text_features(**inputs).detach().cpu().numpy().flatten()
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
        
        # Build path -> dominant mapping
        # We need idx2path to map ids to paths
        index_path = os.path.join(DB_DIR, f"{db_name}_image.json")
        if not os.path.exists(index_path):
            return {}
        
        with open(index_path) as f:
            idx2path = json.load(f)
        idx2path = {int(k): v for k, v in idx2path.items()}
        
        cache = {}
        for i, img_id in enumerate(ids):
            if img_id in idx2path:
                path = idx2path[img_id]
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
    path = urllib.parse.unquote(p)
    data = make_thumbnail_bytes(path)
    if data is None:
        return Response(status=404)
    return Response(data, mimetype="image/jpeg", headers={"Cache-Control": "public, max-age=31536000"})


@app.server.route("/preview")
def preview_endpoint():
    p = request.args.get("p")
    if not p:
        return Response("missing p", status=400)
    path = urllib.parse.unquote(p)

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
    path = urllib.parse.unquote(p)
    
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
    path = urllib.parse.unquote(p)
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
    path = urllib.parse.unquote(p)
    data = make_waveform_png(path)
    if data is None:
        return Response(status=404)
    return Response(data, mimetype="image/png", headers={"Cache-Control": "public, max-age=31536000"})

@app.server.route("/aspec")
def aspec_endpoint():
    p = request.args.get("p")
    if not p:
        return Response("missing p", status=400)
    path = urllib.parse.unquote(p)
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


app.layout = html.Div(
    style={"backgroundColor": "#121212", "color": "white", "padding": "20px", "height": "100vh"},
    children=[
        html.Div(
            [
                dcc.RadioItems(
                    id="mode-select",
                    options=[
                        {"label": "Prompt Search", "value": "prompt"},
                        {"label": "Generate Story", "value": "story"},
                        {"label": "Moodboard", "value": "moodboard"},
                    ],
                    value="prompt",
                    labelStyle={"display": "inline-block", "marginRight": "25px", "fontWeight": "bold"},
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
        dcc.Store(id="story-cache", storage_type="memory"),
        dcc.Store(id="grouped-results", storage_type="memory"),
        dcc.Store(id="carousel-state", storage_type="memory"),
        dcc.Store(id="carousel-order", storage_type="memory"),
        dcc.Store(id="moodboard-store", storage_type="local"),  # Persist moodboard across sessions
        dcc.Store(id="selected-moodboard-image", storage_type="memory"),  # Currently selected reference image
        html.Div(
            [
                html.Div(
                    dcc.Graph(id="scatter-plot", style={"height": "80vh"}),
                    id="scatter-wrapper",
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
                        html.Button("Search", id="main-action-btn", n_clicks=0),
                    ],
                    id="controls-bar",
                    style={"display": "flex", "alignItems": "center", "marginTop": "10px"},
                ),
                # Moodboard controls (hidden by default)
                html.Div(
                    [
                        # Header Row with Upload Zone
                        html.Div(
                            [
                                html.Div("Similarity Search", style={"fontSize": "16px", "fontWeight": "600", "color": "#00bcd4"}),
                                dcc.Upload(
                                    id="external-image-upload",
                                    children=html.Div(
                                        [
                                            html.Span("📁 Drop image here", style={"marginRight": "8px"}),
                                            html.Span("or click to browse", style={"color": "#888", "fontSize": "11px"}),
                                        ],
                                        style={"display": "flex", "alignItems": "center"},
                                    ),
                                    style={
                                        "padding": "8px 16px",
                                        "border": "2px dashed #444",
                                        "borderRadius": "6px",
                                        "backgroundColor": "#252525",
                                        "cursor": "pointer",
                                        "fontSize": "12px",
                                        "transition": "border-color 0.2s",
                                    },
                                    style_active={"borderColor": "#00bcd4"},
                                    accept="image/*",
                                    multiple=False,
                                ),
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
                                        ], style={"marginBottom": "8px"}),
                                        html.Div([
                                            html.Span("Method:", style={"fontSize": "11px", "color": "#888", "marginRight": "6px"}),
                                            dcc.Dropdown(
                                                id="palette-method",
                                                options=[
                                                    {"label": "Histogram", "value": "histogram"},
                                                    {"label": "EMD", "value": "emd"},
                                                    {"label": "Moments", "value": "moments"},
                                                ],
                                                value="histogram",
                                                clearable=False,
                                                style={"width": "100px", "color": "#000", "fontSize": "12px"},
                                            ),
                                        ], style={"display": "flex", "alignItems": "center"}),
                                        html.Div([
                                            html.Span("Colors:", style={"fontSize": "11px", "color": "#888", "marginRight": "6px"}),
                                            dcc.Dropdown(
                                                id="palette-n-colors",
                                                options=[{"label": str(n), "value": n} for n in [4, 8, 12, 16, 24, 32]],
                                                value=16,
                                                clearable=False,
                                                style={"width": "60px", "color": "#000", "fontSize": "12px"},
                                            ),
                                        ], style={"display": "flex", "alignItems": "center", "marginTop": "6px"}),
                                    ],
                                    style={"padding": "10px", "backgroundColor": "#252525", "borderRadius": "8px", "border": "1px solid #333", "flex": "1", "minWidth": "140px"},
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
                                        ], style={"marginBottom": "8px"}),
                                        html.Div([
                                            html.Span("Method:", style={"fontSize": "11px", "color": "#888", "marginRight": "6px"}),
                                            dcc.Dropdown(
                                                id="style-method",
                                                options=[
                                                    {"label": "Gram", "value": "gram"},
                                                    {"label": "Edge", "value": "edge"},
                                                    {"label": "LBP", "value": "lbp"},
                                                ],
                                                value="gram",
                                                clearable=False,
                                                style={"width": "80px", "color": "#000", "fontSize": "12px"},
                                            ),
                                        ], style={"display": "flex", "alignItems": "center"}),
                                    ],
                                    style={"padding": "10px", "backgroundColor": "#252525", "borderRadius": "8px", "border": "1px solid #333", "flex": "1", "minWidth": "140px"},
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
                                    style={"flex": "1", "minWidth": "150px", "padding": "8px", "borderRadius": "4px", "border": "1px solid #444", "backgroundColor": "#2a2a2a", "color": "#fff"},
                                ),
                                html.Div([
                                    html.Span("Results:", style={"fontSize": "11px", "color": "#888", "marginRight": "4px"}),
                                    dcc.Input(
                                        id="moodboard-num",
                                        type="number",
                                        value=50,
                                        min=1,
                                        max=500,
                                        style={"width": "60px", "padding": "6px", "borderRadius": "4px", "border": "1px solid #444", "backgroundColor": "#2a2a2a", "color": "#fff", "textAlign": "center"},
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
                                    style={"padding": "8px 20px", "backgroundColor": "#00bcd4", "color": "#fff", "border": "none", "borderRadius": "4px", "fontWeight": "600", "cursor": "pointer"},
                                ),
                            ],
                            style={"display": "flex", "gap": "12px", "alignItems": "center", "flexWrap": "wrap"},
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
            style={"width": "55%", "display": "inline-block", "verticalAlign": "top"},
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

                # Moodboard gallery (shown in moodboard mode)
                html.Div(
                    [
                        html.Div(
                            [
                                html.Span("📌 Reference Images", style={"fontWeight": "600", "color": "#00bcd4"}),
                                html.Span(" — click to select, then Find Similar", style={"fontSize": "12px", "color": "#888"}),
                            ],
                            style={"marginBottom": "10px"},
                        ),
                        html.Div(id="moodboard-gallery", style={
                            "display": "grid", 
                            "gridTemplateColumns": "repeat(auto-fill, minmax(90px, 1fr))",
                            "gap": "8px",
                            "maxHeight": "300px",
                            "overflowY": "auto",
                            "padding": "4px",
                        }),
                        html.Div(
                            [
                                html.Button(
                                    "Clear References", 
                                    id="clear-moodboard-btn", 
                                    n_clicks=0, 
                                    style={"padding": "6px 14px", "backgroundColor": "#444", "border": "none", "borderRadius": "4px", "color": "#ccc", "fontSize": "12px", "cursor": "pointer"},
                                ),
                                html.Button(
                                    "Save Moodboard", 
                                    id="save-moodboard-btn", 
                                    n_clicks=0, 
                                    style={"padding": "6px 14px", "backgroundColor": "#2a6a4f", "border": "none", "borderRadius": "4px", "color": "#fff", "fontSize": "12px", "cursor": "pointer"},
                                ),
                            ],
                            style={"display": "flex", "gap": "8px", "marginTop": "10px"},
                        ),
                        # Moodboard save folder input
                        dcc.Input(
                            id="moodboard-save-folder",
                            type="text",
                            placeholder="Folder name to save moodboard images...",
                            style={"width": "100%", "marginTop": "8px", "padding": "8px", "borderRadius": "4px", "border": "1px solid #444", "backgroundColor": "#2a2a2a", "color": "#fff"},
                        ),
                        html.Div(id="moodboard-save-confirmation", style={"marginTop": "6px", "fontSize": "12px"}),
                        
                        # Divider
                        html.Hr(style={"margin": "16px 0", "borderColor": "#444"}),
                        
                        # Selection controls for results
                        html.Div("Results Selection", style={"fontWeight": "600", "color": "#888", "fontSize": "12px", "marginBottom": "8px", "textTransform": "uppercase", "letterSpacing": "0.5px"}),
                        html.Div(
                            [
                                html.Button("Select All", id="moodboard-select-all", n_clicks=0, 
                                           style={"padding": "6px 14px", "backgroundColor": "#333", "border": "none", "borderRadius": "4px", "color": "#ccc", "fontSize": "12px", "cursor": "pointer"}),
                                html.Button("Clear All", id="moodboard-clear-all", n_clicks=0,
                                           style={"padding": "6px 14px", "backgroundColor": "#333", "border": "none", "borderRadius": "4px", "color": "#ccc", "fontSize": "12px", "cursor": "pointer"}),
                            ],
                            style={"display": "flex", "gap": "8px", "marginBottom": "10px"},
                        ),
                        html.Button("Save Selected Images", id="moodboard-save-selected", n_clicks=0,
                                   style={"width": "100%", "padding": "8px", "backgroundColor": "#00796b", "border": "none", "borderRadius": "4px", "color": "#fff", "fontWeight": "600", "cursor": "pointer"}),
                        dcc.Input(
                            id="moodboard-results-folder",
                            type="text",
                            placeholder="Folder name to save selected results...",
                            style={"width": "100%", "marginTop": "8px", "padding": "8px", "borderRadius": "4px", "border": "1px solid #444", "backgroundColor": "#2a2a2a", "color": "#fff"},
                        ),
                        html.Div(id="moodboard-results-confirmation", style={"marginTop": "6px", "fontSize": "12px"}),
                    ],
                    id="moodboard-section",
                    style={"display": "none", "marginBottom": "16px", "padding": "14px", 
                           "backgroundColor": "#1e1e1e", "borderRadius": "10px", "border": "1px solid #333"},
                ),

                # Results list
                html.Div(
                    id="image-display",
                    style={"overflowY": "scroll", "overflowX": "hidden", "maxHeight": "80vh"},
                ),

                # Bulk selection buttons (hidden in moodboard mode)
                html.Div(
                    [
                        html.Button("Select All", id="select-all", n_clicks=0),
                        html.Button("Clear All", id="clear-all", n_clicks=0),
                    ],
                    id="bulk-selection-btns",
                    style={"display": "flex", "gap": "8px", "marginTop": "8px"},
                ),

                # Save actions (hidden in moodboard mode)
                html.Div(
                    [
                        html.Button("Save Selected Images", id="save-button", style={"marginTop": "8px"}),
                        dcc.Input(
                            id="save-folder",
                            type="text",
                            placeholder="Enter folder path...",
                            style={"width": "100%", "marginTop": "6px"},
                        ),
                    ],
                    id="save-actions-section",
                ),
                html.Button("Save Story", id="save-story-btn", style={"marginTop": "10px", "display": "none"}),
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
    style={"width": "42%", "display": "inline-block", "paddingLeft": "3%", "verticalAlign": "top"},
),

# hover-thumb unchanged
html.Img(
    id="hover-thumb",
    style={
        "display": "none",
        "position": "fixed",
        "top": "8px",
        "left": "8px",
        "zIndex": 1000,
        "maxWidth": "160px",
        "maxHeight": "120px",
        "border": "2px solid #fff",
        "boxShadow": "0 0 12px #000",
        "backgroundColor": "#000",
        "objectFit": "contain",
    },
)


    ],
)


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
    ],
    Input("mode-select", "value"),
)
def toggle_inputs(mode):
    controls_visible = {"display": "flex", "alignItems": "center", "marginTop": "10px"}
    controls_hidden = {"display": "none"}
    moodboard_controls_visible = {"display": "block", "marginTop": "10px", "padding": "10px", "backgroundColor": "#1a1a1a", "borderRadius": "5px"}
    moodboard_section_visible = {"display": "block", "marginBottom": "12px", "padding": "10px", 
                                  "backgroundColor": "#1e1e1e", "borderRadius": "5px"}
    moodboard_section_hidden = {"display": "none"}
    
    # Column styles
    left_col_normal = {"width": "55%", "display": "inline-block", "verticalAlign": "top"}
    right_col_normal = {"width": "42%", "display": "inline-block", "paddingLeft": "3%", "verticalAlign": "top"}
    
    # For moodboard: small left (just moodboard controls), larger right (results)
    left_col_moodboard = {"width": "30%", "display": "inline-block", "verticalAlign": "top"}
    right_col_moodboard = {"width": "67%", "display": "inline-block", "paddingLeft": "3%", "verticalAlign": "top"}
    
    scatter_visible = {"display": "block"}
    scatter_hidden = {"display": "none"}
    
    bulk_btns_visible = {"display": "flex", "gap": "8px", "marginTop": "8px"}
    save_section_visible = {"display": "block"}
    
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
        )
    else:  # moodboard
        return (
            {"display": "none"},
            {"display": "none"},
            {"display": "none"},
            "Search",
            controls_hidden,
            moodboard_controls_visible,
            moodboard_section_visible,
            scatter_hidden,
            left_col_moodboard,
            right_col_moodboard,
            {"display": "none"},  # hide bulk selection
            {"display": "none"},  # hide save actions
        )


# ─────────────────────────────────────────────────────────────────────────────
# MOODBOARD CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(
    Output("moodboard-store", "data"),
    [
        Input({"type": "add-to-moodboard", "index": ALL}, "n_clicks"),
        Input({"type": "remove-from-moodboard", "index": ALL}, "n_clicks"),
        Input("clear-moodboard-btn", "n_clicks"),
    ],
    State("moodboard-store", "data"),
    prevent_initial_call=True,
)
def update_moodboard(add_clicks, remove_clicks, clear_clicks, current_moodboard):
    """Add or remove images from moodboard."""
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
        Input("palette-n-colors", "value"),
        Input("mode-select", "value"),
    ],
)
def update_moodboard_ref_display(ref_image, n_colors, mode):
    """Update the reference image display in the left column."""
    # Only show in moodboard mode
    if mode != "moodboard" or not ref_image:
        return [], {"display": "none"}
    
    n_colors = n_colors or 16
    qpath = urllib.parse.quote(ref_image)
    
    content = [
        html.Div("REFERENCE IMAGE", style={"color": "#00bcd4", "fontWeight": "600", "fontSize": "11px", "marginBottom": "10px", "textTransform": "uppercase", "letterSpacing": "1px"}),
        html.Img(
            src=f"/preview?p={qpath}&w=600",
            srcSet=f"/preview?p={qpath}&w=400 400w, /preview?p={qpath}&w=600 600w, /preview?p={qpath}&w=900 900w",
            sizes="25vw",
            style={"width": "100%", "borderRadius": "6px", "border": "2px solid #00bcd4", "marginBottom": "8px"},
        ),
        html.Img(
            src=f"/palette?p={qpath}&n={n_colors}&w=300&h=20",
            style={"width": "100%", "borderRadius": "4px"},
        ),
        html.Div(
            os.path.basename(ref_image),
            style={"color": "#888", "fontSize": "10px", "marginTop": "8px", "wordBreak": "break-all"},
        ),
    ]
    
    style = {
        "display": "block", 
        "marginTop": "12px", 
        "padding": "12px", 
        "backgroundColor": "#1a1a1a", 
        "borderRadius": "10px", 
        "border": "1px solid #00bcd4",
        "position": "sticky",
        "top": "10px",
    }
    
    return content, style


@app.callback(
    Output("moodboard-gallery", "children"),
    [
        Input("moodboard-store", "data"),
        Input("selected-moodboard-image", "data"),
    ],
)
def render_moodboard_gallery(moodboard, selected_path):
    """Render clickable thumbnails in the moodboard gallery."""
    moodboard = moodboard or []
    if not moodboard:
        return [html.Div("No images in moodboard. Use Search mode and click '+ Moodboard' to add images.", 
                        style={"color": "#888", "fontStyle": "italic"})]
    
    thumbnails = []
    for path in moodboard:
        qpath = urllib.parse.quote(path)
        is_selected = (path == selected_path)
        border = "3px solid #00bcd4" if is_selected else "2px solid transparent"
        thumbnails.append(
            html.Div([
                # Clickable image area
                html.Div(
                    html.Img(
                        src=f"/thumb?p={qpath}",
                        style={"width": "80px", "height": "80px", "objectFit": "cover", 
                               "borderRadius": "4px", "border": border, "display": "block"},
                    ),
                    id={"type": "moodboard-thumb", "index": path},
                    n_clicks=0,
                    style={"cursor": "pointer"},
                ),
                # X button to remove
                html.Button(
                    "×",
                    id={"type": "remove-from-moodboard", "index": path},
                    n_clicks=0,
                    style={
                        "position": "absolute", "top": "-6px", "right": "-6px",
                        "width": "18px", "height": "18px", "borderRadius": "50%",
                        "backgroundColor": "#ff4444", "color": "white", "border": "none",
                        "fontSize": "12px", "lineHeight": "1", "cursor": "pointer",
                        "padding": "0", "fontWeight": "bold", "zIndex": "10",
                    },
                ),
            ], style={"position": "relative", "display": "inline-block", "margin": "2px"})
        )
    return thumbnails


@app.callback(
    Output("selected-moodboard-image", "data"),
    [
        Input({"type": "moodboard-thumb", "index": ALL}, "n_clicks"),
        Input("external-image-upload", "contents"),
    ],
    [
        State("moodboard-store", "data"),
        State("external-image-upload", "filename"),
    ],
    prevent_initial_call=True,
)
def select_moodboard_image(clicks, upload_contents, moodboard, upload_filename):
    """Select a moodboard image as the reference for similarity search.
    
    Handles both:
    - Clicking on a moodboard thumbnail
    - Drag-and-drop upload of an external image
    """
    triggered = ctx.triggered_id
    
    # Handle moodboard thumbnail click
    if isinstance(triggered, dict) and triggered.get("type") == "moodboard-thumb":
        return triggered["index"]
    
    # Handle external image upload
    if triggered == "external-image-upload" and upload_contents:
        # Decode base64 image data
        content_type, content_string = upload_contents.split(',')
        decoded = base64.b64decode(content_string)
        
        # Get file extension from filename
        ext = os.path.splitext(upload_filename)[1] if upload_filename else ".jpg"
        if not ext:
            ext = ".jpg"
        
        # Save to temp file in arcana/output directory (so /preview endpoint can serve it)
        output_dir = os.path.join(os.path.dirname(__file__), "output", "_external_refs")
        os.makedirs(output_dir, exist_ok=True)
        
        # Create unique filename using hash of content
        content_hash = hashlib.md5(decoded).hexdigest()[:12]
        temp_path = os.path.join(output_dir, f"ext_{content_hash}{ext}")
        
        # Write file
        with open(temp_path, "wb") as f:
            f.write(decoded)
        
        return temp_path
    
    return dash.no_update


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
    output_dir = os.path.join(os.path.dirname(__file__), "output", "moodboards", folder_name)
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
    output_dir = os.path.join(os.path.dirname(__file__), "output", "selections", folder_name)
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
    Input("moodboard-search-btn", "n_clicks"),
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
def moodboard_similarity_search(n_clicks, ref_image, use_palette, palette_method, use_style, style_method, 
                                 n_colors, show_swatches, prompt, num_results, img_size, columns, dataset_value, group_on, sim_thresh):
    """Search for similar images using palette/style, optionally constrained by prompt."""
    if not n_clicks or not ref_image or not dataset_value:
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
    palette_method = palette_method or "histogram"
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


@app.callback(
    [
        Output("save-confirmation", "children", allow_duplicate=True),
        Output("story-cache", "data", allow_duplicate=True),
        Output("image-display", "children", allow_duplicate=True),
    ],
    Input("inject-poetry-btn", "n_clicks"),
    State("story-cache", "data"),
    State("save-folder", "value"),
    State("poetry-strength", "value"),  # <— NEW
    prevent_initial_call="initial_duplicate",
)
def inject_poetry(n_clicks, story_cache, folder, strength_val):

    if not story_cache or "story" not in story_cache:
        return "No story data available.", dash.no_update, dash.no_update

    subfolder = folder or "story"
    output_dir = os.path.join(STORIES_DIR, subfolder, "poetry_injected")
    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained("stabilityai/sd-turbo", torch_dtype=torch.float16, variant="fp16").to(
        device
    )

    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_vae_slicing()
    pipe.safety_checker = None
    pipe.watermark = None

    # strength = 0.72
    strength = 0.72 if strength_val is None else max(0.0, min(1.0, float(strength_val)))

    num_steps = 4
    guidance_scale = 1.0
    negative_prompt = "text, letters, watermark, logo, blurry, low quality"

    updated_story_images = []
    new_image_display = []

    for idx, item in enumerate(story_cache["story"]):
        img_path = resolve_path(item["path"])
        prompt = item["text"]
        init_img = Image.open(img_path).convert("RGB")

        max_width = 1024
        w, h = init_img.size
        if w > max_width:
            scale = max_width / float(w)
            new_w = (max_width // 8) * 8
            new_h = (int(h * scale) // 8) * 8
            init_img = init_img.resize((new_w, new_h), Image.LANCZOS)
        else:
            new_w, new_h = (w // 8) * 8, (h // 8) * 8
            init_img = init_img.resize((new_w, new_h), Image.LANCZOS)

        gen = torch.manual_seed(2222 + idx)

        out_img = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=init_img,
            strength=strength,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=gen,
        ).images[0]

        gen_name = _short_poetry_name(img_path, prompt, idx, ext="png")
        poetry_img_path = os.path.join(output_dir, gen_name)

        # Save (optionally with long-path helper on Windows)
        save_path = _win_longpath(poetry_img_path)  # or just poetry_img_path if not using the helper
        out_img.save(save_path)

        # ⬇️ Put this line here:
        poetry_img_path = os.path.abspath(poetry_img_path)

        # Then build the base64 and append to updated_story_images
        buffered = BytesIO()
        out_img.save(buffered, format="JPEG")
        poetry_img_str = base64.b64encode(buffered.getvalue()).decode()

        updated_story_images.append({
            "text": prompt,
            "path": item["path"],
            "original_img_str": item["img_str"],
            "poetry_img_str": poetry_img_str,
            "poetry_img_path": poetry_img_path,  # now absolute
        })


        # Replace UI image with poetry-injected one
        new_image_display.append(
            html.Div(
                [
                    html.H5(prompt, style={"marginBottom": "4px", "color": "#ffc107"}),
                    html.Img(src=f"data:image/jpeg;base64,{poetry_img_str}", style={"width": "100%", "marginBottom": "10px"}),
                    daq.BooleanSwitch(id={"type": "select-image", "index": item["path"]}, on=True, style={"display": "none"}),
                ],
                style={"marginBottom": "24px", "padding": "10px", "backgroundColor": "#1e1e1e", "borderRadius": "5px"},
            )
        )

    pipe.to("cpu")
    del pipe
    torch.cuda.empty_cache()

    # Update the story_cache with poetry images included
    updated_cache = {"story": updated_story_images, "chunks": story_cache["chunks"]}

    return f"Poetry-injected images saved successfully in {output_dir}.", updated_cache, new_image_display

@lru_cache(maxsize=40000)  # cache reads of files we've already written
def _read_cached_spec(key: str) -> bytes | None:
    f = os.path.join(SPEC_CACHE_DIR, f"{key}.png")
    try:
        with open(f, "rb") as fh:
            return fh.read()
    except Exception:
        return None

def _write_cached_spec(key: str, data: bytes) -> None:
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
        device = "cuda" if torch.cuda.is_available() else "cpu"
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

    # Nothing selected yet
    if not dataset_value:
        return [], go.Figure(), {"display": "none"}, {}, [], {}, []

    # Parse "<name>::<dim>::<modality>" (backward compat to ::dim)
    try:
        parts = (dataset_value or "").split("::")
        if len(parts) == 3:
            latent_name, dim, modality = parts[0], int(parts[1]), parts[2]
        else:
            latent_name, dim, modality = parts[0], int(parts[1]), "image"
    except Exception as e:
        print(f"[update_images] bad dataset value={dataset_value!r}: {e}")
        return [], go.Figure(), {"display": "none"}, {}, [], {}, []
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
                fig.add_trace(
                    go.Scatter(
                        x=xs,
                        y=ys,
                        mode="lines+markers",
                        line=dict(color="gold", width=4),
                        marker=dict(size=14, color="gold"),
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
            images.append(  # unchanged wrapper...

                html.Div(
                    [
                        html.H5(img["text"], style={"marginBottom": "4px", "color": "#ffc107"}),
                        media,
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
            highlighted_df = df.loc[[r[0] for r in results]]
            print(f"[DEBUG] Highlighted DataFrame: {highlighted_df.shape[0]} rows")
            # ...inside update_images(), in the PROMPT block, after highlighted_df = ...
            # after: highlighted_df = df.loc[[r[0] for r in results]]

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

                # NOTE: use go.Scatter (SVG) so it sits above the scattergl canvas
                fig.add_trace(
                    go.Scatter(
                        x=xs, y=ys,
                        mode="markers",
                        marker=dict(
                            symbol="x",          # or "x-thin-open"
                            size=20,
                            color="#33C3F0",
                            line=dict(width=2),
                        ),
                        name="Search Results",
                        hoverinfo="skip",
                        showlegend=True,
                        cliponaxis=False,       # keeps the X strokes visible at edges
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
    # Hide hover thumb in moodboard mode (scatter is hidden)
    if mode == "moodboard":
        return "", {"display": "none"}
    
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
                style = {
                    "display": "block",
                    "position": "fixed",
                    "top": "100px",
                    "left": "100px",
                    "width": "128px",
                    "height": "128px",
                    "border": "2px solid #fff",
                    "zIndex": 1000,
                    "boxShadow": "0 0 12px #000",
                }
                return thumb_url, style
    except Exception as e:
        print("[hover-thumb] skipped due to:", e)
    return dash.no_update, {"display": "none"}




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
        save_dir = os.path.join(SELECTIONS_DIR, subfolder)
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
                img = cv2.imread(full_path)
                if img is not None:
                    cv2.imwrite(os.path.join(save_dir, safe_name), img)
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
        save_dir = os.path.join(STORIES_DIR, subfolder)
        poetry_dir = os.path.join(save_dir, "poetry_injected")
        original_dir = os.path.join(save_dir, "original")
        os.makedirs(poetry_dir, exist_ok=True)
        os.makedirs(original_dir, exist_ok=True)

        n_saved = 0
        if story_cache and "story" in story_cache:
            for i, item in enumerate(story_cache["story"]):
                full_img_path = resolve_path(item["path"])
                original_img = cv2.imread(full_img_path)
                if original_img is not None:
                    cv2.imwrite(os.path.join(original_dir, f"{i:02d}_original.jpg"), original_img)
                    n_saved += 1
                poetry_img_path = item.get("poetry_img_path")
                if poetry_img_path and os.path.exists(poetry_img_path):
                    poetry_img = cv2.imread(poetry_img_path)
                    if poetry_img is not None:
                        cv2.imwrite(os.path.join(poetry_dir, f"{i:02d}_poetry.jpg"), poetry_img)
                        n_saved += 1

            with open(os.path.join(save_dir, "story.txt"), "w", encoding="utf-8") as f:
                for i, chunk in enumerate(story_cache.get("chunks", [])):
                    f.write(f"{i+1}. {chunk}\n")

            msg = f"Story and {n_saved} images (original + poetry-injected) saved successfully to {save_dir}."
        else:
            msg = "No story to save."
    return msg


def main():
    app.run(debug=False)


if __name__ == "__main__":
    main()

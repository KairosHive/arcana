import dash
from dash import dcc, html, Input, Output, State, ctx
import plotly.express as px
import pandas as pd
import cv2
import base64
import os
import pickle
import torch
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
import librosa
from matplotlib import pyplot as plt





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
                    options=[{"label": "Prompt Search", "value": "prompt"}, {"label": "Generate Story", "value": "story"}],
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
        html.Div(
            [
                dcc.Graph(id="scatter-plot", style={"height": "80vh"}),
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
            ],
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

                # Results list
                html.Div(
                    id="image-display",
                    style={"overflowY": "scroll", "overflowX": "hidden", "maxHeight": "80vh"},
                ),

                # Bulk selection buttons
                html.Div(
                    [
                        html.Button("Select All", id="select-all", n_clicks=0),
                        html.Button("Clear All", id="clear-all", n_clicks=0),
                    ],
                    style={"display": "flex", "gap": "8px", "marginTop": "8px"},
                ),

                # Save actions
                html.Button("Save Selected Images", id="save-button", style={"marginTop": "8px"}),
                dcc.Input(
                    id="save-folder",
                    type="text",
                    placeholder="Enter folder path...",
                    style={"width": "100%", "marginTop": "6px"},
                ),
                html.Button("Save Story", id="save-story-btn", style={"marginTop": "10px", "display": "none"}),
                html.Div(id="save-confirmation", style={"marginTop": "10px"}),
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
    ],
    Input("mode-select", "value"),
)
def toggle_inputs(mode):
    if mode == "prompt":
        return (
            {"display": "block", "width": "60%", "marginRight": "10px"},
            {"display": "block", "width": "15%", "marginRight": "10px"},
            {"display": "none"},
            "Search",
        )
    else:
        return (
            {"display": "none"},
            {"display": "none"},
            {"display": "block", "width": "70%", "height": "70px", "marginRight": "10px"},
            "Generate Story",
        )


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
        fig = px.scatter(df, x="x", y="y", **scatter_kwargs)

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
                        marker=dict(size=10, color="gold"),
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
            if is_3d:
                fig.add_trace(
                    go.Scatter3d(
                        x=highlighted_df["x"],
                        y=highlighted_df["y"],
                        z=highlighted_df["z"],
                        mode="markers",
                        marker=dict(color="cyan", size=8, symbol="x"),
                        name="Search Results",
                    )
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=highlighted_df["x"],
                        y=highlighted_df["y"],
                        mode="markers",
                        marker=dict(color="cyan", size=12, symbol="x"),
                        name="Search Results",
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
                            daq.BooleanSwitch(id={"type": "select-image", "index": first}, on=False),
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
                            daq.BooleanSwitch(id={"type": "select-image", "index": f"group::{g['gid']}"}, on=False),
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
                    daq.BooleanSwitch(id={"type": "select-image", "index": media_path}, on=False),
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
        State("spec-toggle", "on"),  # <— NEW
    ],
    prevent_initial_call=True,
)
def nav_carousel(left_clicks, right_clicks, groups, order, car_state, spec_on):

    groups = groups or []
    order = order or []
    car_groups = {g["gid"]: g for g in groups if len(g.get("paths", [])) > 1}

    if not order:
        return car_state or {}, [], [], [], []

    car_state = dict(car_state or {})
    for gid in car_groups:
        car_state.setdefault(gid, 0)
    car_state = {gid: idx for gid, idx in car_state.items() if gid in car_groups}

    trig = ctx.triggered_id
    if isinstance(trig, dict) and trig.get("type") in ("left", "right"):
        gid = trig.get("gid")
        g = car_groups.get(gid)
        if g:
            n = len(g["paths"])
            cur = car_state.get(gid, 0)
            cur = (cur - 1) % n if trig["type"] == "left" else (cur + 1) % n
            car_state[gid] = cur

    srcs, srcsets, counters, audios = [], [], [], []
    for gid in order:
        g = car_groups.get(gid)
        if not g:
            srcs.append(dash.no_update); srcsets.append(dash.no_update)
            counters.append(dash.no_update); audios.append(dash.no_update)
            continue
        paths = g["paths"]
        cur = car_state.get(gid, 0) % len(paths)
        qp = urllib.parse.quote(paths[cur])

        # If it's an image carousel, we use /preview and set srcset; if audio, we use /awave + /audio
        is_audio = os.path.splitext(paths[0])[1].lower() in AUDIO_EXTS
        if is_audio:
            preview_ep = "/aspec" if spec_on else "/awave"
            srcs.append(f"{preview_ep}?p={qp}")
            srcsets.append(dash.no_update)
            audios.append(f"/audio?p={qp}")
        else:
            srcs.append(f"/preview?p={qp}&w=900")
            srcsets.append(f"/preview?p={qp}&w=600 600w, /preview?p={qp}&w=900 900w, /preview?p={qp}&w=1400 1400w")
            audios.append(dash.no_update)


        counters.append(f"{cur+1}/{len(paths)}")

    return car_state, srcs, srcsets, counters, audios


@app.callback(
    Output("hover-thumb", "src"),
    Output("hover-thumb", "style"),
    Input("scatter-plot", "hoverData"),
    State("dataset-dropdown", "value"),
    State("spec-toggle", "on"),
)
def update_hover_thumb(hoverData, dataset_value, spec_on):
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

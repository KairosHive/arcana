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


torch.set_grad_enabled(False)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
LATENTS_DIR = os.path.join(APP_ROOT, "latents")
DB_DIR = os.path.join(APP_ROOT, "databases")

IMAGES_ROOT = os.path.abspath(os.path.join(APP_ROOT, "..", "images"))

OUTPUT_DIR = os.path.join(APP_ROOT, "output")
STORIES_DIR = os.path.join(OUTPUT_DIR, "stories")
SELECTIONS_DIR = os.path.join(OUTPUT_DIR, "selections")

os.makedirs(STORIES_DIR, exist_ok=True)
os.makedirs(SELECTIONS_DIR, exist_ok=True)


# ------------- FILE DISCOVERY HELPERS -------------

from functools import lru_cache


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
    latent_pattern = re.compile(r"latent_space_(.+)_(\d+)d\.pkl$")
    latent_map = {}
    for fname in os.listdir(latent_dir):
        m = latent_pattern.match(fname)
        if m:
            name, dim = m.group(1), m.group(2)
            latent_map.setdefault(name, []).append(dim)
    # Now find intersection with databases
    db_pattern = re.compile(r"index_(.+)\.pkl$")
    db_names = {m.group(1) for fname in os.listdir(db_dir) if (m := db_pattern.match(fname))}
    options = []
    for name, dims in latent_map.items():
        if name in db_names:
            for dim in sorted(dims):
                label = f"{name} ({dim}D)"
                value = f"{name}::{dim}"
                options.append({"label": label, "value": value})
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


def load_data(name, n_dim=2):
    latent_path = os.path.join(LATENTS_DIR, f"latent_space_{name}_{n_dim}d.pkl")
    df = pd.read_pickle(latent_path)

    # normalize schema / dtypes
    if "label" in df.columns:
        df["label"] = df["label"].astype(str)
    if "path" in df.columns:
        df["path"] = df["path"].astype(str)
    for col in ("x", "y", "z"):
        if col in df.columns:
            df[col] = df[col].astype("float32")  # lighter plots

    return df.reset_index(drop=True)


def load_index(name):
    index_name = os.path.join(DB_DIR, f"index_{name}.pkl")
    with open(index_name, "rb") as f:
        index, idx2path = pickle.load(f)
    index = Index.restore(index)
    return index, idx2path


# ------------- CLIP MODEL LOAD ONCE -------------
model = CLIPModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K", torch_dtype=torch.float16)
# model.eval().to("cuda")
model.eval().to("cpu")
processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")


def search(index, idx2path, query, n):
    # Move model to GPU for the search
    # model.to("cuda")
    inputs = processor.tokenizer(query, return_tensors="pt")
    vec = model.get_text_features(**inputs).detach().cpu().numpy().flatten()
    # model.to("cpu")                     # <<<<<<<< BACK TO CPU
    # torch.cuda.empty_cache()            # <<<<<<<< RELEASE VRAM
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
        html.Div(
            [
                # Place the button above images!
                html.Button("Inject Poetry", id="inject-poetry-btn", style={"marginBottom": "10px", "display": "none"}),
                # your results list
                html.Div(
                    [
                        html.Label("Group twins", style={"marginRight": "8px"}),
                        daq.BooleanSwitch(id="group-similar", on=True, color="#00bcd4"),
                        html.Label("  distance ≤", style={"marginLeft": "12px", "marginRight": "6px"}),
                        dcc.Input(
                            id="sim-thresh",
                            type="number",
                            value=0.08,
                            step=0.01,
                            min=0.0,
                            max=0.5,
                            style={"width": "90px", "color": "#000"},
                        ),
                    ],
                    style={"margin": "8px 0 6px 0"},
                ),
                html.Div(id="image-display", style={"overflowY": "scroll", "overflowX": "hidden", "maxHeight": "80vh"}),
                # ⬇️ ADD THIS BLOCK HERE ⬇️
                html.Div(
                    [
                        html.Button("Select All", id="select-all", n_clicks=0, style={"marginRight": "8px"}),
                        html.Button("Clear All", id="clear-all", n_clicks=0),
                    ],
                    style={"marginTop": "6px"},
                ),
                # ⬆️ ADD THIS BLOCK HERE ⬆️
                html.Button("Save Selected Images", id="save-button"),
                dcc.Input(
                    id="save-folder",
                    type="text",
                    placeholder="Enter folder path...",
                    style={"width": "100%", "marginTop": "5px"},
                ),
                html.Button(
                    "Save Story", id="save-story-btn", style={"marginTop": "10px", "marginLeft": "0px", "display": "none"}
                ),
                html.Div(id="save-confirmation", style={"marginTop": "10px"}),
            ],
            style={"width": "42%", "display": "inline-block", "paddingLeft": "3%", "verticalAlign": "top"},
        ),
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
        ),
    ],
)


@app.callback(
    [
        Output("search-box", "style"),
        Output("num-images", "style"),
        Output("story-box", "style"),
        Output("main-action-btn", "children"),
        Output("inject-poetry-btn", "style"),
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
            {"display": "none"},
        )
    else:
        return (
            {"display": "none"},
            {"display": "none"},
            {"display": "block", "width": "70%", "height": "70px", "marginRight": "10px"},
            "Generate Story",
            {"display": "block", "marginTop": "10px"},
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
    prevent_initial_call="initial_duplicate",
)
def inject_poetry(n_clicks, story_cache, folder):
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

    strength = 0.72
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

        safe_prompt = "".join(c if c.isalnum() or c in " _-" else "_" for c in prompt)[:60]
        gen_name = f"{idx:02d}_{os.path.splitext(os.path.basename(img_path))[0]}_{safe_prompt}_poetry.png"
        poetry_img_path = os.path.join(output_dir, gen_name)
        out_img.save(poetry_img_path)

        # Encode the poetry image directly for UI
        buffered = BytesIO()
        out_img.save(buffered, format="JPEG")
        poetry_img_str = base64.b64encode(buffered.getvalue()).decode()

        updated_story_images.append(
            {
                "text": prompt,
                "path": item["path"],
                "original_img_str": item["img_str"],
                "poetry_img_str": poetry_img_str,
                "poetry_img_path": poetry_img_path,
            }
        )

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


@app.callback(
    [
        Output("image-display", "children"),
        Output("scatter-plot", "figure"),
        Output("save-story-btn", "style"),
        Output("story-cache", "data"),
        Output("grouped-results", "data"),
        Output("carousel-state", "data"),
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
    ],
)
def update_images(
    n_action, clickData, mode, dataset_value, search_value, num_images, relayoutData, story_value, group_on, sim_thresh
):
    # Nothing selected yet
    if not dataset_value:
        return [], go.Figure(), {"display": "none"}, {}, [], {}

    # Parse "<name>::<dim>" defensively
    try:
        latent_name, dim = dataset_value.split("::", 1)
        n_dim = int(dim)
    except Exception as e:
        print(f"[update_images] bad dataset value={dataset_value!r}: {e}")
        return [], go.Figure(), {"display": "none"}, {}, [], {}

    db_name = latent_name

    # Load coordinates only; DO NOT load the search index yet
    df = load_data(latent_name, n_dim=n_dim)

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
        index, idx2path = load_index(db_name)
        print("[DEBUG] STORY mode triggered")
        story_chunks = [chunk.strip() for chunk in story_value.split("\n") if chunk.strip()]
        print(f"[DEBUG] Story chunks: {story_chunks}")
        story_images = []
        for i, chunk in enumerate(story_chunks):
            results = search(index, idx2path, chunk, 1)
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
            images.append(
                html.Div(
                    [
                        html.H5(img["text"], style={"marginBottom": "4px", "color": "#ffc107"}),
                        html.Img(
                            src=f"/preview?p={qpath}&w=900",
                            srcSet=(
                                f"/preview?p={qpath}&w=600 600w, "
                                f"/preview?p={qpath}&w=900 900w, "
                                f"/preview?p={qpath}&w=1400 1400w"
                            ),
                            sizes="(max-width: 900px) 90vw, 42vw",
                            style={"width": "100%", "marginBottom": "10px"},
                        ),
                    ],
                    style={"marginBottom": "24px", "padding": "10px", "backgroundColor": "#1e1e1e", "borderRadius": "5px"},
                )
            )
        show_save_story = {"display": "block", "marginTop": "10px"}
        story_cache = {"story": story_images, "chunks": story_chunks}
        return images, fig, show_save_story, story_cache, groups_store, car_state_store

    # --- PROMPT mode with grouping / carousel ---
    if mode == "prompt" and trigger == "main-action-btn" and search_value:
        print("[DEBUG] PROMPT mode triggered")
        index, idx2path = load_index(db_name)
        results = search(index, idx2path, search_value, num_images)
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
        print(f"[DEBUG] Carousel state: {car_state}")
        cards = []
        for g in groups:
            n = len(g["paths"])
            print(f"[DEBUG] Group {g['gid']} has {n} paths: {g['paths']}")
            first = g["paths"][0]
            qpath = urllib.parse.quote(first)
            if n == 1:
                cards.append(
                    html.Div(
                        [
                            html.Img(
                                src=f"/preview?p={qpath}&w=900",
                                srcSet=(
                                    f"/preview?p={qpath}&w=600 600w, "
                                    f"/preview?p={qpath}&w=900 900w, "
                                    f"/preview?p={qpath}&w=1400 1400w"
                                ),
                                sizes="(max-width: 900px) 90vw, 42vw",
                                style={"width": "100%", "marginBottom": "10px"},
                            ),
                            daq.BooleanSwitch(id={"type": "select-image", "index": first}, on=False),
                            html.Span(" (no twins)", style={"marginLeft": "10px", "opacity": 0.7}),
                        ],
                        style={"marginBottom": "20px", "padding": "10px", "backgroundColor": "#1e1e1e", "borderRadius": "5px"},
                    )
                )
            else:
                cards.append(
                    html.Div(
                        [
                            # --- overlay container ---
                            html.Div(
                                [
                                    html.Img(
                                        id={"type": "carousel-img", "gid": g["gid"]},
                                        src=f"/preview?p={qpath}&w=900",
                                        srcSet=(
                                            f"/preview?p={qpath}&w=600 600w, "
                                            f"/preview?p={qpath}&w=900 900w, "
                                            f"/preview?p={qpath}&w=1400 1400w"
                                        ),
                                        sizes="(max-width: 900px) 90vw, 42vw",
                                        style={
                                            "width": "100%",
                                            "display": "block",
                                            "marginBottom": "10px",
                                            "borderRadius": "5px",
                                        },
                                    ),
                                    # left arrow
                                    html.Button(
                                        "◀",
                                        id={"type": "left", "gid": g["gid"]},
                                        n_clicks=0,
                                        style={
                                            "position": "absolute",
                                            "left": "8px",
                                            "top": "50%",
                                            "transform": "translateY(-50%)",
                                            "backgroundColor": "rgba(0,0,0,0.6)",
                                            "color": "#fff",
                                            "border": "none",
                                            "borderRadius": "9999px",
                                            "width": "36px",
                                            "height": "36px",
                                            "zIndex": 2,
                                            "cursor": "pointer",
                                        },
                                    ),
                                    # right arrow
                                    html.Button(
                                        "▶",
                                        id={"type": "right", "gid": g["gid"]},
                                        n_clicks=0,
                                        style={
                                            "position": "absolute",
                                            "right": "8px",
                                            "top": "50%",
                                            "transform": "translateY(-50%)",
                                            "backgroundColor": "rgba(0,0,0,0.6)",
                                            "color": "#fff",
                                            "border": "none",
                                            "borderRadius": "9999px",
                                            "width": "36px",
                                            "height": "36px",
                                            "zIndex": 2,
                                            "cursor": "pointer",
                                        },
                                    ),
                                ],
                                # key: make the arrows overlay the image
                                style={"position": "relative", "overflow": "hidden"},
                            ),
                            html.Div(
                                id={"type": "carousel-counter", "gid": g["gid"]},
                                children=f"1/{n}",
                                style={"textAlign": "center", "margin": "4px 0 8px 0", "opacity": 0.8},
                            ),
                            daq.BooleanSwitch(id={"type": "select-image", "index": f"group::{g['gid']}"}, on=False),
                            html.Span(f" twins: {n}", style={"marginLeft": "10px", "opacity": 0.7}),
                        ],
                        style={
                            "marginBottom": "20px",
                            "padding": "10px",
                            "backgroundColor": "#1e1e1e",
                            "borderRadius": "5px",
                            "overflowX": "hidden",  # defensive: no horizontal scrollbar
                        },
                    )
                )

        print(f"[DEBUG] Returning {len(cards)} cards")
        return cards, fig, {"display": "none"}, {}, groups, car_state

    # --- Scatter click (unchanged) ---
    if trigger == "scatter-plot" and clickData:
        pt = clickData["points"][0]
        custom = pt.get("customdata") or []
        image_path = custom[0] if custom else None
        if not image_path:
            # No 'path' available for this dataset; just return the fig without cards
            return images, fig, {"display": "none"}, {}, [], {}

        qpath = urllib.parse.quote(image_path)
        images.append(
            html.Div(
                [
                    html.Img(
                        src=f"/preview?p={qpath}&w=900",
                        srcSet=(
                            f"/preview?p={qpath}&w=600 600w, "
                            f"/preview?p={qpath}&w=900 900w, "
                            f"/preview?p={qpath}&w=1400 1400w"
                        ),
                        sizes="(max-width: 900px) 90vw, 42vw",
                        style={"width": "100%", "marginBottom": "10px"},
                    ),
                    daq.BooleanSwitch(id={"type": "select-image", "index": image_path}, on=False),
                ],
                style={"marginBottom": "20px", "padding": "10px", "backgroundColor": "#1e1e1e", "borderRadius": "5px"},
            )
        )
        return images, fig, {"display": "none"}, {}, [], {}

    # keep camera on pan/zoom
    if is_3d and relayoutData and "scene.camera" in relayoutData:
        fig.update_layout(scene_camera=relayoutData["scene.camera"])

    return images, fig, show_save_story, story_cache, groups_store, car_state_store


@app.callback(
    [
        Output("carousel-state", "data", allow_duplicate=True),
        Output({"type": "carousel-img", "gid": ALL}, "src"),
        Output({"type": "carousel-img", "gid": ALL}, "srcSet"),
        Output({"type": "carousel-counter", "gid": ALL}, "children"),
    ],
    [
        Input({"type": "left", "gid": ALL}, "n_clicks"),
        Input({"type": "right", "gid": ALL}, "n_clicks"),
    ],
    [
        State("carousel-state", "data"),
        State("grouped-results", "data"),
    ],
    prevent_initial_call=True,
)
def nav_carousel(left_clicks, right_clicks, car_state, groups):
    if not groups:
        return car_state, [], [], []

    if car_state is None:
        car_state = {}

    # Only groups that actually render a carousel component
    car_groups = [g for g in groups if len(g.get("paths", [])) > 1]

    # If there are no carousel components in the layout, return empty lists
    if not car_groups:
        return car_state, [], [], []

    trig = ctx.triggered_id
    if isinstance(trig, dict) and trig.get("type") in ("left", "right"):
        gid = trig.get("gid")
        group = next((g for g in car_groups if g["gid"] == gid), None)
        if group:
            n = len(group["paths"])
            cur = car_state.get(gid, 0)
            cur = (cur - 1) % n if trig["type"] == "left" else (cur + 1) % n
            car_state[gid] = cur

    srcs, srcsets, counters = [], [], []
    for g in car_groups:
        gid = g["gid"]
        paths = g["paths"]
        cur = car_state.get(gid, 0) % len(paths)
        p = paths[cur]
        qp = urllib.parse.quote(p)
        srcs.append(f"/preview?p={qp}&w=900")
        srcsets.append(f"/preview?p={qp}&w=600 600w, " f"/preview?p={qp}&w=900 900w, " f"/preview?p={qp}&w=1400 1400w")
        counters.append(f"{cur+1}/{len(paths)}")

    return car_state, srcs, srcsets, counters


@app.callback(Output("hover-thumb", "src"), Output("hover-thumb", "style"), Input("scatter-plot", "hoverData"))
def update_hover_thumb(hoverData):
    try:
        if hoverData and "points" in hoverData:
            pt = hoverData["points"][0]
            custom = pt.get("customdata") or []
            if custom:
                image_path = custom[0]
                thumb_url = "/thumb?p=" + urllib.parse.quote(image_path)
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
            full_img_path = resolve_path(path)
            basename = os.path.basename(full_img_path)
            img = cv2.imread(full_img_path)
            if img is not None:
                cv2.imwrite(os.path.join(save_dir, basename), img)
                n_saved += 1
            else:
                print(f"[ERROR] Could not load image: {full_img_path}")
        msg = f"{n_saved} images saved successfully to {save_dir}."

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

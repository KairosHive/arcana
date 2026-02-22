# db.py — unified latent DB builder for images (CLIP) and audio (CLAP)
# Supports:
#   - Image datasets indexed with CLIP (text<->image)
#   - Audio datasets indexed with CLAP (text<->audio)
# Saves:
#   databases/index_<name>_<modality>.pkl
#   latents/latent_space_<name>_<modality>_<n_components>D.pkl

import os
import cv2
import math
import pickle
import argparse
from glob import glob
import hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing


import numpy as np
import torch
from tqdm import tqdm
import pandas as pd

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

from usearch.index import Index
from transformers import CLIPModel, CLIPProcessor, ClapModel, ClapProcessor

from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import random


# Optional audio backends
try:
    import torchaudio
except Exception:  # pragma: no cover
    torchaudio = None
try:
    import soundfile as sf
except Exception:  # pragma: no cover
    sf = None

# Optional palette/style feature extractors
try:
    from .palette import extract_all_palette_features
    PALETTE_AVAILABLE = True
except ImportError:
    try:
        from palette import extract_all_palette_features
        PALETTE_AVAILABLE = True
    except ImportError:
        PALETTE_AVAILABLE = False
try:
    from .style import extract_all_style_features
    STYLE_AVAILABLE = True
except ImportError:
    try:
        from style import extract_all_style_features
        STYLE_AVAILABLE = True
    except ImportError:
        STYLE_AVAILABLE = False

torch.set_grad_enabled(False)

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
script_root = os.path.dirname(os.path.abspath(__file__))
IMAGES_ROOT = os.path.abspath(os.path.join(script_root, "..", "images"))
ASSETS_DIR = os.path.join(script_root, "assets")
DEFAULT_LABELS = {
    "image": os.path.join(ASSETS_DIR, "labels_image.txt"),
    "audio": os.path.join(ASSETS_DIR, "labels_audio.txt"),
}


db_dir = os.path.join(script_root, "databases")
latents_dir = os.path.join(script_root, "latents")
os.makedirs(db_dir, exist_ok=True)
os.makedirs(latents_dir, exist_ok=True)

# --------------------------------------------------------------------------------------
# File types
# --------------------------------------------------------------------------------------
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
CLIP_MODEL_ID = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
CLAP_MODEL_ID = "laion/clap-htsat-fused"


def is_image(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in IMAGE_EXTS

def is_audio(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in AUDIO_EXTS

# ---------- NEW: text encoder + label helpers ----------


def _row_norm(X: np.ndarray) -> np.ndarray:
    X = X.astype(np.float32, copy=False)
    X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    return X

def choose_k(
    X: np.ndarray,
    k_min: int = 2,
    k_max: int = 20,
    metric: str = "silhouette",   # "silhouette" | "calinski" | "davies"
    sample_size: int = 5000,
    random_state: int = 0,
) -> tuple[int, dict[int, float]]:
    """
    Normalize X once, optionally subsample for speed, then score KMeans for k in [k_min, k_max].
    Returns (best_k, scores_by_k).
    """
    Xn = _row_norm(X)
    if Xn.shape[0] > sample_size:
        idx = random.Random(random_state).sample(range(Xn.shape[0]), sample_size)
        Xs = Xn[idx]
    else:
        Xs = Xn

    scores = {}
    best_k, best_val = None, None

    for k in range(max(2, k_min), max(2, k_max) + 1):
        try:
            km = KMeans(n_clusters=k, random_state=random_state, n_init="auto").fit(Xs)
            labels = km.labels_
            if metric == "silhouette":
                val = silhouette_score(Xs, labels)
                better = (best_val is None) or (val > best_val)
            elif metric == "calinski":
                val = calinski_harabasz_score(Xs, labels)
                better = (best_val is None) or (val > best_val)
            elif metric == "davies":
                val = davies_bouldin_score(Xs, labels)
                better = (best_val is None) or (val < best_val)  # lower is better
            else:
                raise ValueError(f"Unknown metric: {metric}")
            scores[k] = float(val)
            if better:
                best_k, best_val = k, val
        except Exception as e:
            print(f"[auto-k] k={k} skipped: {e}")

    if best_k is None:
        best_k = max(2, k_min)
    return best_k, scores


def _text2vec(text: str, modality: str) -> np.ndarray:
    """Route text encoding to the right model for this dataset's modality."""
    if modality == "image":
        return txt2vec_clip(text)
    elif modality == "audio":
        return txt2vec_clap(text)
    else:
        raise ValueError(f"Unsupported modality for text2vec: {modality}")

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32); b = b.astype(np.float32)
    na = np.linalg.norm(a) + 1e-8
    nb = np.linalg.norm(b) + 1e-8
    return float((a @ b) / (na * nb))

def _load_label_dict(path_or_inline: str | None) -> dict[str, list[str]]:
    """
    Accepts:
      - path to a JSON or CSV
        * JSON: {"labelA": ["term1", "term2"], "labelB": ["..."]}
        * CSV : columns `label,term` (multiple rows per label)
      - inline comma list "rain,wind,thunder"  -> {"rain":["rain"], ...}
      - None -> empty dict
    """
    if not path_or_inline:
        return {}

    # Inline comma list?
    if ("," in path_or_inline) and (not os.path.exists(path_or_inline)):
        labels = [x.strip() for x in path_or_inline.split(",") if x.strip()]
        return {lab: [lab] for lab in labels}

    # File path
    p = os.path.abspath(path_or_inline)
    if not os.path.exists(p):
        print(f"[WARN] Label dictionary path not found: {path_or_inline}")
        return {}

    ext = os.path.splitext(p)[1].lower()
    if ext in {".json"}:
        import json
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        # normalize to dict[str, list[str]]
        out: dict[str, list[str]] = {}
        for k, v in data.items():
            if isinstance(v, str):
                out[k] = [v]
            elif isinstance(v, list):
                out[k] = [str(t) for t in v]
        return out

    elif ext in {".csv"}:
        import csv
        out: dict[str, list[str]] = {}
        with open(p, "r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                lab = str(row.get("label", "")).strip()
                term = str(row.get("term", "")).strip()
                if lab and term:
                    out.setdefault(lab, []).append(term)
        return out

    else:
        print(f"[WARN] Unsupported label dict extension: {ext}")
        return {}

# ---------- NEW: label list + embeddings cache ----------

_LABEL_MEM_CACHE: dict[str, tuple[list[str], np.ndarray]] = {}

def _read_label_list(src: str | None) -> tuple[list[str], str]:
    """
    Returns (labels, cache_key_base).
    - If src is a file path (.txt): read all non-empty lines.
    - If src is inline comma list: split by comma.
    - If None/empty: return ([], "empty").
    cache_key_base is used to form a stable on-disk cache filename.
    """
    if not src:
        return [], "empty"

    if os.path.exists(src):
        # TXT file (one label per line)
        p = os.path.abspath(src)
        with open(p, "rb") as f:
            raw = f.read()
        # content-based hash so we invalidate when file content changes
        h = hashlib.md5(raw).hexdigest()[:12]
        text = raw.decode("utf-8", errors="ignore")
        labels = [ln.strip() for ln in text.splitlines() if ln.strip()]
        return labels, f"file:{p}|md5:{h}"
    else:
        # inline comma list
        labels = [x.strip() for x in src.split(",") if x.strip()]
        h = hashlib.md5(",".join(labels).encode("utf-8")).hexdigest()[:12]
        return labels, f"inline:{h}"

def _text2vec(text: str, modality: str) -> np.ndarray:
    if modality == "image":
        return txt2vec_clip(text)
    elif modality == "audio":
        return txt2vec_clap(text)
    else:
        raise ValueError(f"Unsupported modality: {modality}")

def _encode_label_matrix(
    labels: list[str],
    modality: str,
    cache_base: str,
    cache_dir: str = db_dir,  # reuse your databases/ dir
) -> tuple[list[str], np.ndarray]:
    """
    Returns (labels, M) where M is (L, D) of L2-normalized label embeddings.
    Uses both in-memory and on-disk caches so we never re-encode unchanged labels.
    """
    if not labels:
        return [], np.zeros((0, 1), dtype=np.float32)

    model_id = CLIP_MODEL_ID if modality == "image" else CLAP_MODEL_ID
    cache_key = f"{modality}|{model_id}|{cache_base}"
    if cache_key in _LABEL_MEM_CACHE:
        return _LABEL_MEM_CACHE[cache_key]

    # on-disk cache filename
    model_tag = hashlib.md5(model_id.encode("utf-8")).hexdigest()[:8]
    disk_hash = hashlib.md5(cache_key.encode("utf-8")).hexdigest()[:12]
    cache_path = os.path.join(cache_dir, f"label_cache_{modality}_{disk_hash}_{model_tag}.pkl")

    # Try load disk cache
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                saved = pickle.load(f)
            saved_labels, M = saved["labels"], saved["embeddings"]
            if saved_labels == labels and isinstance(M, np.ndarray):
                # ensure normalized
                M = M.astype(np.float32)
                M /= (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)
                _LABEL_MEM_CACHE[cache_key] = (saved_labels, M)
                return saved_labels, M
        except Exception as e:
            print(f"[WARN] failed to load label cache ({cache_path}): {e}")

    # Encode (first try vectorized batch; fallback to loop)
    vecs = []
    B = 64  # batch size for text encoding
    for i in range(0, len(labels), B):
        batch = labels[i : i + B]
        try:
            # fast path: encode each text with the proper encoder
            if modality == "image":
                # CLIP tokenizer can take batch
                model, processor = load_clip()
                toks = processor.tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
                with torch.no_grad():
                    v = model.get_text_features(toks.input_ids.to(model.device)).detach().cpu().float().numpy()
                vecs.append(v)
            else:
                # CLAP text batch
                model, proc = load_clap()
                inputs = proc(text=batch, return_tensors="pt", padding=True)
                for k in inputs:
                    inputs[k] = inputs[k].to(model.device)
                with torch.no_grad():
                    try:
                        v = model.get_text_features(**inputs)
                    except AttributeError:
                        v = model(**inputs).text_embeds
                vecs.append(v.detach().cpu().float().numpy())
        except Exception:
            # robust fallback: per-item
            for t in batch:
                try:
                    vecs.append(_text2vec(t, modality)[None, :])
                except Exception as e:
                    print(f"[WARN] text2vec failed for '{t}': {e}")

    if not vecs:
        return [], np.zeros((0, 1), dtype=np.float32)

    M = np.concatenate(vecs, axis=0).astype(np.float32)
    # L2 normalize for cosine via dot
    M /= (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)

    # Save disk cache
    try:
        with open(cache_path, "wb") as f:
            pickle.dump({"labels": labels, "embeddings": M}, f)
    except Exception as e:
        print(f"[WARN] failed to write label cache ({cache_path}): {e}")

    _LABEL_MEM_CACHE[cache_key] = (labels, M)
    return labels, M


def _infer_cluster_names_from_matrix(
    item_vecs: np.ndarray,         # (N, D), raw or normalized
    cluster_ids: np.ndarray,       # (N,)
    label_texts: list[str],        # L strings
    label_mat_norm: np.ndarray,    # (L, D) L2-normalized
) -> tuple[dict[int, str], dict[int, float]]:
    """
    Centroid of each cluster -> nearest label via cosine (dot with normalized rows).
    Returns cid->label and cid->score.
    """
    if label_mat_norm.shape[0] == 0:
        return {}, {}

    # normalize items once
    X = item_vecs.astype(np.float32)
    X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

    cid2name, cid2score = {}, {}

    for cid in sorted(set(cluster_ids.tolist())):
        mask = (cluster_ids == cid)
        if not np.any(mask):
            continue
        centroid = X[mask].mean(axis=0)
        centroid /= (np.linalg.norm(centroid) + 1e-8)
        # cosine to all labels -> argmax
        scores = label_mat_norm @ centroid  # (L,)
        j = int(np.argmax(scores))
        cid2name[cid] = label_texts[j]
        cid2score[cid] = float(scores[j])
    return cid2name, cid2score

# --------------------------------------------------------------------------------------
# Lazy model loaders
# --------------------------------------------------------------------------------------
_CLIP = {"model": None, "proc": None}
_CLAP = {"model": None, "proc": None}

def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

def load_clip(device: str | None = None):
    device = device or _device()
    if _CLIP["model"] is None:
        m = CLIPModel.from_pretrained(CLIP_MODEL_ID)
        if device == "cuda":
            m = m.to("cuda").half()
        else:
            m = m.to("cpu")
        m.eval()
        p = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
        _CLIP.update(model=m, proc=p)
    return _CLIP["model"], _CLIP["proc"]

def load_clap(device: str | None = None):
    device = device or _device()
    if _CLAP["model"] is None:
        m = ClapModel.from_pretrained(CLAP_MODEL_ID)
        # Keep CLAP in FP32 (BN layers are happier; avoids dtype mismatch)
        if device == "cuda":
            m = m.to("cuda")   # <-- no .half()
        else:
            m = m.to("cpu")
        m.eval()
        p = ClapProcessor.from_pretrained(CLAP_MODEL_ID)
        _CLAP.update(model=m, proc=p)
    return _CLAP["model"], _CLAP["proc"]


# --------------------------------------------------------------------------------------
# Encoders
# --------------------------------------------------------------------------------------
def img2vec_clip(image_bgr: np.ndarray) -> np.ndarray:
    """Input: BGR uint8 image (cv2)."""
    model, processor = load_clip()
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    px = processor(images=[rgb], return_tensors="pt").pixel_values.to(model.device)
    with torch.no_grad():
        vec = model.get_image_features(px).squeeze().detach().cpu().float().numpy()
    return vec

def txt2vec_clip(text: str) -> np.ndarray:
    model, processor = load_clip()
    toks = processor.tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        vec = model.get_text_features(toks.input_ids.to(model.device)).squeeze().detach().cpu().float().numpy()
    return vec

def read_audio_mono(
    path: str,
    target_sr: int = 48000,
    seconds: int | None = None,   # None = variable length
    pad: bool = False,            # only used if seconds is not None
):
    """Load mono float32, resample, optional crop/pad to 'seconds'."""
    if torchaudio is not None:
        wav, sr = torchaudio.load(path)  # (ch, n)
        wav = wav.mean(dim=0).numpy()
    else:
        wav, sr = sf.read(path, always_2d=False)
        if wav.ndim == 2:
            wav = wav.mean(axis=1)

    # resample
    if sr != target_sr:
        if torchaudio is not None:
            wav = torchaudio.functional.resample(torch.from_numpy(wav), sr, target_sr).numpy()
        else:
            import librosa
            wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)

    # optional crop/pad
    if seconds is not None:
        max_len = int(target_sr * seconds)
        if wav.shape[0] > max_len:
            wav = wav[:max_len]               # crop
        elif pad:
            wav = np.pad(wav, (0, max_len - wav.shape[0]))  # right-pad

    return wav.astype(np.float32), target_sr


def aud2vec_clap(audio_np: np.ndarray, sr: int) -> np.ndarray:
    model, processor = load_clap()
    inputs = processor(audios=audio_np, sampling_rate=sr, return_tensors="pt")
    for k in inputs:
        inputs[k] = inputs[k].to(model.device)
    with torch.no_grad():
        # get_audio_features exists in recent transformers; fallback to forward
        try:
            emb = model.get_audio_features(**inputs)
        except AttributeError:
            emb = model(**inputs).audio_embeds
        vec = emb.squeeze().detach().cpu().float().numpy()
    return vec

def txt2vec_clap(text: str) -> np.ndarray:
    model, processor = load_clap()
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    for k in inputs:
        inputs[k] = inputs[k].to(model.device)
    with torch.no_grad():
        try:
            emb = model.get_text_features(**inputs)
        except AttributeError:
            emb = model(**inputs).text_embeds
        vec = emb.squeeze().detach().cpu().float().numpy()
    return vec

# --------------------------------------------------------------------------------------
# Index building
# --------------------------------------------------------------------------------------
def build(glob_path: str, index_path: str, batch_size: int = 32, modality: str = "image") -> tuple[Index, dict]:
    """
    Build a cosine index for the given media files.
    modality: "image" or "audio"
    """
    # discover files
    all_paths = glob(glob_path, recursive=True)
    paths = [p for p in all_paths if (is_image(p) if modality == "image" else is_audio(p))]
    print(f"[INFO] Found {len(paths)} {modality} files to index.")
    if not paths:
        raise SystemExit("No files found for indexing.")

    # probe ndim from first vector
    if modality == "image":
        probe = cv2.imread(paths[0])
        if probe is None:
            raise SystemExit(f"Failed to read first image: {paths[0]}")
        v0 = img2vec_clip(probe)
    else:
        # PROBE (before creating the index)
        a, sr = read_audio_mono(paths[0], target_sr=48000, seconds=None, pad=False)
        v0 = aud2vec_clap(a, sr)


    ndim = int(v0.shape[-1])
    index = Index(ndim=ndim, metric="cos")
    idx2path: dict[int, str] = {}

    if modality == "image":
        for batch_start in tqdm(range(0, len(paths), batch_size), desc="Indexing images"):
            batch_paths = paths[batch_start : batch_start + batch_size]
            imgs = []
            ok_paths = []
            for p in batch_paths:
                im = cv2.imread(p)
                if im is None:
                    print(f"[WARN] bad image, skipping: {p}")
                    continue
                imgs.append(im)
                ok_paths.append(p)
            if not ok_paths:
                continue
            # Encode batch
            model, processor = load_clip()
            px = processor(images=[cv2.cvtColor(x, cv2.COLOR_BGR2RGB) for x in imgs],
                           return_tensors="pt").pixel_values.to(model.device)
            with torch.no_grad():
                vecs = model.get_image_features(px).detach().cpu().float().numpy()
            for i, vec in enumerate(vecs):
                gid = batch_start + i
                index.add(gid, vec)
                idx2path[gid] = os.path.abspath(ok_paths[i])

    else:  # audio
        # MAIN LOOP
        for i, p in enumerate(tqdm(paths, desc="Indexing audio")):
            try:
                a, sr = read_audio_mono(p, target_sr=48000, seconds=None, pad=False)
                vec = aud2vec_clap(a, sr)
                index.add(i, vec)
                idx2path[i] = os.path.abspath(p)
            except Exception as e:
                print(f"[WARN] failed on {p}: {e}")


    with open(index_path, "wb") as f:
        pickle.dump((index.save(), idx2path), f)

    return index, idx2path


# --------------------------------------------------------------------------------------
# Additional feature extraction (palette, style)
# --------------------------------------------------------------------------------------
def _extract_palette_worker(args):
    """Worker function for parallel palette extraction (must be module-level for pickling)."""
    i, path = args
    try:
        # Import inside worker to avoid issues with multiprocessing
        try:
            from arcana.palette import extract_all_palette_features as _extract
        except ImportError:
            from palette import extract_all_palette_features as _extract
        feats = _extract(path)
        return (i, feats, None)
    except Exception as e:
        return (i, None, str(e))


def extract_additional_features(
    idx2path: dict[int, str],
    name: str,
    features: list[str],
    include_gram: bool = True,
    compact_gram: bool = True,
    gram_pca_dims: int = 0,
    n_workers: int = 1,
) -> dict[str, str]:
    """
    Extract palette and/or style features for all indexed images.
    
    Args:
        idx2path: Mapping from index ID to absolute image path
        name: Project name for output files
        features: List of features to extract ("palette", "style")
        include_gram: Whether to include Gram matrix in style features
        compact_gram: Use 2 VGG layers (~41k dims) instead of 4 (~174k dims)
        gram_pca_dims: If > 0, compress Gram features to this many dims via PCA
        n_workers: Number of parallel workers for CPU-bound features (palette, style w/o gram)
        
    Returns:
        Dict of feature_type -> output_path for saved .npz files
    """
    output_paths = {}
    
    # Get ordered list of paths (by index key)
    sorted_ids = sorted(idx2path.keys())
    paths = [idx2path[i] for i in sorted_ids]
    n_images = len(paths)
    
    # --- Palette features ---
    if "palette" in features:
        if not PALETTE_AVAILABLE:
            print("[WARN] palette.py not found, skipping palette features")
        else:
            print(f"\n[INFO] Extracting palette features for {n_images} images (workers={n_workers})...")
            histograms = [None] * n_images
            dominant_colors = [None] * n_images
            color_moments = [None] * n_images
            valid_mask = [False] * n_images
            
            work_items = list(enumerate(paths))
            
            if n_workers > 1:
                with ProcessPoolExecutor(max_workers=n_workers) as executor:
                    futures = {executor.submit(_extract_palette_worker, item): item for item in work_items}
                    for future in tqdm(as_completed(futures), total=len(work_items), desc="Palette features"):
                        try:
                            i, feats, err = future.result(timeout=5)  # 5s timeout per image
                            if feats is not None:
                                histograms[i] = feats['histogram']
                                dominant_colors[i] = feats['dominant']
                                color_moments[i] = feats['moments']
                                valid_mask[i] = True
                            elif err:
                                print(f"[WARN] Palette failed on {paths[i]}: {err}")
                        except TimeoutError:
                            item = futures[future]
                            print(f"[WARN] Palette timeout on {paths[item[0]]}")
                        except Exception as e:
                            item = futures[future]
                            print(f"[WARN] Palette error on {paths[item[0]]}: {e}")
            else:
                for item in tqdm(work_items, desc="Palette features"):
                    i, feats, err = _extract_palette_worker(item)
                    if feats is not None:
                        histograms[i] = feats['histogram']
                        dominant_colors[i] = feats['dominant']
                        color_moments[i] = feats['moments']
                        valid_mask[i] = True
                    elif err:
                        print(f"[WARN] Palette failed on {paths[i]}: {err}")
            
            valid_indices = [i for i, v in enumerate(valid_mask) if v]
            valid_ids = [sorted_ids[i] for i in valid_indices]
            
            if valid_ids:
                palette_path = os.path.join(db_dir, f"features_{name}_palette.npz")
                np.savez_compressed(
                    palette_path,
                    ids=np.array(valid_ids, dtype=np.int32),
                    histogram=np.stack([histograms[i] for i in valid_indices]).astype(np.float32),
                    dominant=np.stack([dominant_colors[i] for i in valid_indices]).astype(np.float32),
                    moments=np.stack([color_moments[i] for i in valid_indices]).astype(np.float32),
                )
                print(f"[OK] Saved palette features to {palette_path}")
                print(f"     histogram: {histograms[valid_indices[0]].shape} x {len(valid_ids)}")
                print(f"     dominant:  {dominant_colors[valid_indices[0]].shape} x {len(valid_ids)}")
                print(f"     moments:   {color_moments[valid_indices[0]].shape} x {len(valid_ids)}")
                output_paths["palette"] = palette_path
    
    # --- Style features ---
    if "style" in features:
        if not STYLE_AVAILABLE:
            print("[WARN] style.py not found, skipping style features")
        else:
            print(f"\n[INFO] Extracting style features for {n_images} images...")
            if include_gram:
                print(f"     Gram mode: {'compact (~41k dims)' if compact_gram else 'full (~174k dims)'}")
                if gram_pca_dims > 0:
                    print(f"     Will compress to {gram_pca_dims} dims via PCA")
            
            edge_histograms = []
            lbp_textures = []
            gram_features = [] if include_gram else None
            valid_ids = []
            
            for idx, path in zip(tqdm(sorted_ids, desc="Style features"), paths):
                try:
                    feats = extract_all_style_features(path, include_gram=include_gram, compact_gram=compact_gram)
                    edge_histograms.append(feats['edge_histogram'])
                    lbp_textures.append(feats['texture_lbp'])
                    if include_gram and feats.get('gram') is not None:
                        gram_features.append(feats['gram'])
                    valid_ids.append(idx)
                except Exception as e:
                    print(f"[WARN] Style failed on {path}: {e}")
            
            if valid_ids:
                style_path = os.path.join(db_dir, f"features_{name}_style.npz")
                save_dict = {
                    'ids': np.array(valid_ids, dtype=np.int32),
                    'edge_histogram': np.stack(edge_histograms).astype(np.float32),
                    'texture_lbp': np.stack(lbp_textures).astype(np.float32),
                }
                if include_gram and gram_features:
                    gram_array = np.stack(gram_features).astype(np.float32)
                    
                    # Optional PCA compression
                    if gram_pca_dims > 0 and gram_array.shape[0] > gram_pca_dims:
                        from sklearn.decomposition import PCA
                        print(f"     Compressing Gram: {gram_array.shape[1]} → {gram_pca_dims} dims via PCA...")
                        pca = PCA(n_components=gram_pca_dims, random_state=42)
                        gram_array = pca.fit_transform(gram_array)
                        save_dict['gram_pca_components'] = pca.components_.astype(np.float32)
                        save_dict['gram_pca_mean'] = pca.mean_.astype(np.float32)
                        print(f"     Variance retained: {pca.explained_variance_ratio_.sum():.1%}")
                    
                    save_dict['gram'] = gram_array.astype(np.float32)
                
                np.savez_compressed(style_path, **save_dict)
                print(f"[OK] Saved style features to {style_path}")
                print(f"     edge_histogram: {edge_histograms[0].shape} x {len(edge_histograms)}")
                print(f"     texture_lbp:    {lbp_textures[0].shape} x {len(lbp_textures)}")
                if include_gram and gram_features:
                    print(f"     gram:           {save_dict['gram'].shape[1]} dims x {len(gram_features)} images")
                output_paths["style"] = style_path
    
    return output_paths


# --------------------------------------------------------------------------------------
# Feature-based search (palette, style)
# --------------------------------------------------------------------------------------
def load_palette_features(name: str) -> dict | None:
    """Load palette features from .npz file."""
    path = os.path.join(db_dir, f"features_{name}_palette.npz")
    if not os.path.exists(path):
        return None
    data = np.load(path)
    return {
        'ids': data['ids'],
        'histogram': data['histogram'],
        'dominant': data['dominant'],
        'moments': data['moments'],
        'path': path,
    }


def load_style_features(name: str) -> dict | None:
    """Load style features from .npz file."""
    path = os.path.join(db_dir, f"features_{name}_style.npz")
    if not os.path.exists(path):
        return None
    data = np.load(path)
    result = {
        'ids': data['ids'],
        'edge_histogram': data['edge_histogram'],
        'texture_lbp': data['texture_lbp'],
        'path': path,
    }
    if 'gram' in data:
        result['gram'] = data['gram']
    return result


def search_by_palette(
    query_image,
    name: str,
    idx2path: dict[int, str],
    method: str = "histogram",
    n_colors: int = 10,
    top_k: int = 20,
) -> list[tuple[str, float]]:
    """
    Search for similar images by palette.
    
    Args:
        query_image: Path to query image or BGR numpy array
        name: Project name (to load features)
        idx2path: Index ID to path mapping
        method: "histogram" (cosine), "moments" (euclidean), or "emd" (Earth Mover's)
        n_colors: Number of dominant colors for EMD
        top_k: Number of results to return
        
    Returns:
        List of (path, score) tuples, sorted by relevance
    """
    try:
        from .palette import (
            extract_all_palette_features,
            histogram_similarity,
            moments_distance,
            emd_palette_distance,
        )
    except ImportError:
        from palette import (
            extract_all_palette_features,
            histogram_similarity,
            moments_distance,
            emd_palette_distance,
        )
    
    # Load index features
    features = load_palette_features(name)
    if features is None:
        raise FileNotFoundError(f"No palette features found for '{name}'")
    
    # Extract query features
    query_feats = extract_all_palette_features(query_image)
    
    ids = features['ids']
    results = []
    
    if method == "histogram":
        # Cosine similarity - higher is better
        query_hist = query_feats['histogram'].reshape(1, -1)
        db_hists = features['histogram']
        # Normalize for cosine
        query_norm = query_hist / (np.linalg.norm(query_hist) + 1e-8)
        db_norm = db_hists / (np.linalg.norm(db_hists, axis=1, keepdims=True) + 1e-8)
        sims = (db_norm @ query_norm.T).flatten()
        for i, sim in enumerate(sims):
            results.append((idx2path[ids[i]], float(sim)))
        results.sort(key=lambda x: -x[1])  # Higher = better
        
    elif method == "moments":
        # Euclidean distance - lower is better
        query_mom = query_feats['moments']
        for i, db_mom in enumerate(features['moments']):
            dist = float(np.linalg.norm(query_mom - db_mom))
            results.append((idx2path[ids[i]], dist))
        results.sort(key=lambda x: x[1])  # Lower = better
        
    elif method == "emd":
        # Earth Mover's Distance - lower is better
        query_dom = query_feats['dominant']
        for i, db_dom in enumerate(features['dominant']):
            dist = emd_palette_distance(query_dom, db_dom, n_colors=n_colors)
            results.append((idx2path[ids[i]], float(dist)))
        results.sort(key=lambda x: x[1])  # Lower = better
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return results[:top_k]


def search_by_style(
    query_image,
    name: str,
    idx2path: dict[int, str],
    method: str = "edge",
    top_k: int = 20,
) -> list[tuple[str, float]]:
    """
    Search for similar images by style.
    
    Args:
        query_image: Path to query image or BGR numpy array
        name: Project name (to load features)
        idx2path: Index ID to path mapping
        method: "edge", "lbp", or "gram"
        top_k: Number of results to return
        
    Returns:
        List of (path, score) tuples, sorted by similarity (higher is better)
    """
    try:
        from .style import (
            extract_all_style_features,
            edge_histogram_similarity,
            texture_lbp_similarity,
            gram_similarity,
        )
    except ImportError:
        from style import (
            extract_all_style_features,
            edge_histogram_similarity,
            texture_lbp_similarity,
            gram_similarity,
        )
    
    # Load index features
    features = load_style_features(name)
    if features is None:
        raise FileNotFoundError(f"No style features found for '{name}'")
    
    # Extract query features
    include_gram = (method == "gram")
    query_feats = extract_all_style_features(query_image, include_gram=include_gram)
    
    ids = features['ids']
    results = []
    
    if method == "edge":
        query_vec = query_feats['edge_histogram'].reshape(1, -1)
        db_vecs = features['edge_histogram']
        # Cosine similarity
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
        db_norm = db_vecs / (np.linalg.norm(db_vecs, axis=1, keepdims=True) + 1e-8)
        sims = (db_norm @ query_norm.T).flatten()
        for i, sim in enumerate(sims):
            results.append((idx2path[ids[i]], float(sim)))
            
    elif method == "lbp":
        query_vec = query_feats['texture_lbp'].reshape(1, -1)
        db_vecs = features['texture_lbp']
        # Cosine similarity
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
        db_norm = db_vecs / (np.linalg.norm(db_vecs, axis=1, keepdims=True) + 1e-8)
        sims = (db_norm @ query_norm.T).flatten()
        for i, sim in enumerate(sims):
            results.append((idx2path[ids[i]], float(sim)))
            
    elif method == "gram":
        if 'gram' not in features:
            raise ValueError("Gram features not available in index (was --no_gram used?)")
        query_vec = query_feats['gram'].reshape(1, -1)
        db_vecs = features['gram']
        # Cosine similarity
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
        db_norm = db_vecs / (np.linalg.norm(db_vecs, axis=1, keepdims=True) + 1e-8)
        sims = (db_norm @ query_norm.T).flatten()
        for i, sim in enumerate(sims):
            results.append((idx2path[ids[i]], float(sim)))
    else:
        raise ValueError(f"Unknown method: {method}")
    
    results.sort(key=lambda x: -x[1])  # Higher similarity = better
    return results[:top_k]


def search_combined(
    query_image,
    name: str,
    idx2path: dict[int, str],
    weights: dict[str, float] | None = None,
    top_k: int = 20,
) -> list[tuple[str, float]]:
    """
    Combined search using multiple features with weighted scoring.
    
    Args:
        query_image: Path to query image or BGR numpy array
        name: Project name
        idx2path: Index ID to path mapping
        weights: Dict of method -> weight, e.g. {"histogram": 0.5, "edge": 0.3, "gram": 0.2}
                 Default: {"histogram": 0.4, "edge": 0.3, "lbp": 0.3}
        top_k: Number of results to return
        
    Returns:
        List of (path, combined_score) tuples
    """
    if weights is None:
        weights = {"histogram": 0.4, "edge": 0.3, "lbp": 0.3}
    
    # Normalize weights
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}
    
    # Collect scores per path
    path_scores: dict[str, float] = {}
    
    for method, weight in weights.items():
        if weight == 0:
            continue
            
        try:
            if method in ("histogram", "moments", "emd"):
                results = search_by_palette(query_image, name, idx2path, method=method, top_k=len(idx2path))
            elif method in ("edge", "lbp", "gram"):
                results = search_by_style(query_image, name, idx2path, method=method, top_k=len(idx2path))
            else:
                print(f"[WARN] Unknown method '{method}', skipping")
                continue
                
            # Normalize scores to [0, 1] for this method
            if results:
                scores = [r[1] for r in results]
                min_s, max_s = min(scores), max(scores)
                range_s = max_s - min_s if max_s > min_s else 1.0
                
                for path, score in results:
                    norm_score = (score - min_s) / range_s
                    # For distance methods (moments, emd), invert so higher = better
                    if method in ("moments", "emd"):
                        norm_score = 1.0 - norm_score
                    if path not in path_scores:
                        path_scores[path] = 0.0
                    path_scores[path] += weight * norm_score
                    
        except FileNotFoundError as e:
            print(f"[WARN] {e}")
        except Exception as e:
            print(f"[WARN] Error in {method}: {e}")
    
    # Sort by combined score
    results = [(path, score) for path, score in path_scores.items()]
    results.sort(key=lambda x: -x[1])
    return results[:top_k]


# --------------------------------------------------------------------------------------
# Latent space (TSNE + kmeans labels for coloring)
# --------------------------------------------------------------------------------------
def latent_space(
    index: Index,
    idx2path: dict[int, str],
    n_components: int = 2,
    modality: str = "image",
    n_clusters: int = 10,
    label_texts: list[str] | None = None,
    label_mat_norm: np.ndarray | None = None,
):
    vecs = []
    paths = []
    for kid in tqdm(idx2path.keys(), desc="Collecting vectors"):
        vecs.append(index.get(kid))
        paths.append(os.path.abspath(idx2path[kid]))
    vecs = np.asarray(vecs, dtype=np.float32)

    # ---- t-SNE only for visualization ----
    perplexity = int(np.clip(math.ceil(len(vecs) / 10), 5, 30))
    tsne = TSNE(n_components=n_components, perplexity=perplexity, init="pca", learning_rate="auto")
    coords = tsne.fit_transform(vecs)

    # ---- choose k (optional) & cluster on ORIGINAL embeddings ----
    X_for_kmeans = _row_norm(vecs)
    if n_clusters is None or n_clusters <= 0:
        # pull globals via closure or pass args in; here we read env-like defaults
        k_min  = int(os.getenv("ARCANA_K_MIN", "2"))
        k_max  = int(os.getenv("ARCANA_K_MAX", "20"))
        metric = os.getenv("ARCANA_K_METRIC", "silhouette")
        best_k, scores = choose_k(X_for_kmeans, k_min=k_min, k_max=k_max, metric=metric)
        print(f"[auto-k] selected k={best_k} via {metric} in [{k_min},{k_max}]  scores={scores}")
        n_clusters = int(best_k)

    try:
        km = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(X_for_kmeans)
        cluster_ids = km.labels_.astype(int)
    except Exception:
        cluster_ids = np.zeros(len(coords), dtype=int)


    # Label clusters if we have a label matrix
    inferred = []
    if label_texts and label_mat_norm is not None and label_mat_norm.shape[0] > 0:
        cid2name, _ = _infer_cluster_names_from_matrix(vecs, cluster_ids, label_texts, label_mat_norm)
        inferred = [cid2name.get(int(c), "") for c in cluster_ids]
    else:
        inferred = [""] * len(cluster_ids)

    return coords, paths, cluster_ids, inferred



# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Latent Space Builder for Image/Audio Datasets")
    parser.add_argument("--imgs_path", type=str, required=True,
                        help="Folder or glob of media (images or audio). Recursive if a directory.")
    parser.add_argument("--name", type=str, required=True, help="Project name (used in filenames).")
    parser.add_argument("--n_components", type=int, default=2, choices=[2, 3],
                        help="Latent space dimensionality (2 or 3).")
    parser.add_argument("--modality", type=str, choices=["image", "audio"], default="image",
                        help="Which encoder/indexer to use.")
    parser.add_argument("--labels", type=str, default=None,
                        help="TXT path (one label per line) or inline comma list: 'rain,wind,thunder'.")
    parser.add_argument("--k", type=int, default=0,
    help="KMeans cluster count. Use 0 for auto.")
    parser.add_argument("--k_min", type=int, default=2,
        help="Auto-k: minimum k to consider.")
    parser.add_argument("--k_max", type=int, default=20,
        help="Auto-k: maximum k to consider.")
    parser.add_argument("--k_metric", type=str, choices=["silhouette","calinski","davies"],
        default="silhouette", help="Auto-k selection metric.")
    parser.add_argument("--features", type=str, default="clip",
        help="Comma-separated features to extract: clip,palette,style (default: clip). "
             "palette: LAB histogram, dominant colors, color moments. "
             "style: edge histogram, LBP texture, Gram matrix.")
    parser.add_argument("--no_gram", action="store_true",
        help="Skip Gram matrix extraction (faster, but less accurate style matching).")
    parser.add_argument("--full_gram", action="store_true",
        help="Use full Gram (4 VGG layers, ~174k dims). Default is compact (2 layers, ~41k dims).")
    parser.add_argument("--gram_pca", type=int, default=512,
        help="Compress Gram features to N dims via PCA (0=no compression). Default: 512.")
    parser.add_argument("--reuse_index", action="store_true",
        help="Skip CLIP extraction if index already exists. Only extract palette/style features.")
    parser.add_argument("--workers", type=int, default=0,
        help="Number of parallel workers for palette/style extraction (0=auto, uses CPU count).")

    return parser.parse_args()

def main():
    args = parse_args()

    def _to_glob(p: str) -> str:
        # if user already provided a glob, use it as-is
        if any(ch in p for ch in "*?[]"):
            return p
        p_abs = os.path.abspath(p)
        return os.path.join(p_abs, "**", "*") if os.path.isdir(p_abs) else p_abs

    media_path = args.imgs_path
    glob_arg = _to_glob(media_path)


    index_name  = os.path.join(db_dir,     f"index_{args.name}_{args.modality}.pkl")
    latent_name = os.path.join(latents_dir, f"latent_space_{args.name}_{args.modality}_{args.n_components}d.pkl")

    print("path to index:       ", index_name)
    print("path to latent space:", latent_name)
    print("search path:         ", glob_arg)
    print("modality:            ", args.modality)

    # Prepare label candidates (TXT or inline), then get cached embeddings
    label_src = args.labels
    if not label_src:
        # pick default by modality if file exists
        cand = DEFAULT_LABELS.get(args.modality)
        if cand and os.path.exists(cand):
            print(f"[INFO] Using default labels: {cand}")
            label_src = cand

    label_texts, cache_base = _read_label_list(label_src)
    if label_texts:
        label_texts, label_mat = _encode_label_matrix(label_texts, args.modality, cache_base)
    else:
        label_mat = np.zeros((0, 1), dtype=np.float32)


    # Check if we can reuse existing index
    if args.reuse_index and os.path.exists(index_name):
        print(f"[INFO] Reusing existing index: {index_name}")
        with open(index_name, "rb") as f:
            saved_index, idx2path = pickle.load(f)
        index = Index.restore(saved_index)
        print(f"[INFO] Loaded {len(idx2path)} indexed paths.")
    else:
        index, idx2path = build(glob_arg, index_name, modality=args.modality)
    
    # --- Extract additional features (palette, style) ---
    feature_list = [f.strip().lower() for f in args.features.split(",")]
    if args.modality == "image" and any(f in feature_list for f in ["palette", "style"]):
        print(f"\n[INFO] Features requested: {feature_list}")
        additional_features = [f for f in feature_list if f in ("palette", "style")]
        n_workers = args.workers if args.workers > 0 else multiprocessing.cpu_count()
        feature_paths = extract_additional_features(
            idx2path=idx2path,
            name=args.name,
            features=additional_features,
            include_gram=(not args.no_gram),
            compact_gram=(not args.full_gram),
            gram_pca_dims=args.gram_pca,
            n_workers=n_workers,
        )
        for ftype, fpath in feature_paths.items():
            print(f"  {ftype}: {fpath}")
    
    if args.k <= 0:
        os.environ["ARCANA_K_MIN"] = str(args.k_min)
        os.environ["ARCANA_K_MAX"] = str(args.k_max)
        os.environ["ARCANA_K_METRIC"] = args.k_metric

    coords, paths, cluster_ids, inferred_names = latent_space(
        index=index,
        idx2path=idx2path,
        n_components=args.n_components,
        modality=args.modality,
        n_clusters=(0 if args.k <= 0 else int(args.k)),
        label_texts=label_texts,
        label_mat_norm=label_mat,
    )

    if args.n_components == 2:
        df = pd.DataFrame(coords, columns=["x", "y"])
    else:
        df = pd.DataFrame(coords, columns=["x", "y", "z"])
    df["path"] = paths
    df["cluster_id"] = cluster_ids.astype(int)
    # prefer inferred; fallback to "C<id>" if empty
    df["label"] = [name if name else f"C{int(cid)}" for name, cid in zip(inferred_names, cluster_ids)]

    df.to_pickle(latent_name)
    print(f"[OK] Saved latent DataFrame to {latent_name}")

if __name__ == "__main__":
    main()

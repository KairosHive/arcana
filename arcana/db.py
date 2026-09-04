# db.py — unified latent DB builder for images (CLIP) and audio (CLAP)
# Supports:
#   - Image datasets indexed with CLIP (text<->image)
#   - Audio datasets indexed with CLAP (text<->audio)
# Saves:
#   databases/index_<name>_<modality>.pkl
#   latents/latent_space_<name>_<modality>_<n_components>D.pkl

import os
import sys
import cv2
try:
    from .cvio import imread_unicode, imread_for_encoder, imwrite_unicode
except ImportError:
    from cvio import imread_unicode, imread_for_encoder, imwrite_unicode
import math
import pickle
import argparse
from glob import glob
import hashlib
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing
try:
    from . import gpu as _gpu
except ImportError:
    import gpu as _gpu



import numpy as np
import torch
from tqdm import tqdm
import pandas as pd

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

from usearch.index import Index
# transformers is imported inside load_clip/load_clap: importing it costs ~5 s and
# is pure waste for anything that only reads an existing index.

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
try:
    from . import paths as _paths
except ImportError:  # running as a loose script
    import paths as _paths

script_root = _paths.APP_ROOT
ASSETS_DIR = _paths.ASSETS_DIR
DEFAULT_LABELS = {
    "image": os.path.join(ASSETS_DIR, "labels_image.txt"),
    "audio": os.path.join(ASSETS_DIR, "labels_audio.txt"),
}

# Resolved through paths.py so a read-only install still imports. Directories are
# created at first write (see main()), never at import time.
db_dir = _paths.subdir("databases")
latents_dir = _paths.subdir("latents")
IMAGES_ROOT = (_paths.media_roots() or [os.path.abspath(os.path.join(script_root, "..", "images"))])[0]

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

def auto_k(n_items: int) -> int:
    """
    A sensible number of clusters for a collection of this size.

    Automatic k used to be chosen purely by silhouette score over k in [2, 20],
    and on CLIP embeddings that reliably returns 2. Silhouette rewards a few
    well-separated blobs, and in a 1024-dimensional space almost any collection
    splits cleanly in half -- so a 246-photo library came back as two clusters
    named "Portrait" and "Street", with every texture, building and landscape
    forced into one or the other. The names were not wrong so much as
    meaningless: a cluster averaging 120 unrelated pictures has no nearest
    label worth printing.

    k ~ sqrt(n/2) is the usual rule of thumb and behaves sensibly across the
    range this app sees:

        30 photos   ->  4        2,000  -> 24 (capped)
        246         -> 11        9,359  -> 24 (capped)

    The floor of 4 keeps tiny folders from collapsing to a single name. The cap
    of 24 exists because the label vocabulary is only 100 words: past that,
    clusters start sharing names, which reads as a bug.
    """
    if n_items <= 0:
        return 2
    if n_items < 24:
        return max(2, min(4, n_items))
    return int(max(4, min(24, round((n_items / 2.0) ** 0.5))))


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

def _text2vec(text: str, modality: str, model_id: str | None = None) -> np.ndarray:
    if modality == "image":
        return txt2vec_clip(text, model_id=model_id)
    elif modality == "audio":
        return txt2vec_clap(text)
    else:
        raise ValueError(f"Unsupported modality: {modality}")

def _encode_label_matrix(
    labels: list[str],
    modality: str,
    cache_base: str,
    cache_dir: str = db_dir,  # reuse your databases/ dir
    model_id: str | None = None,
) -> tuple[list[str], np.ndarray]:
    """
    Returns (labels, M) where M is (L, D) of L2-normalized label embeddings.
    Uses both in-memory and on-disk caches so we never re-encode unchanged labels.
    """
    if not labels:
        return [], np.zeros((0, 1), dtype=np.float32)

    # The caller's encoder wins. The cache key already includes the model id,
    # so switching models produces a separate cache rather than reusing
    # embeddings of the wrong dimension.
    if model_id is None:
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
                # CLIP tokenizer can take batch.
                # model_id matters: cluster names are found by comparing these
                # label embeddings against image centroids, so both must come
                # from the same encoder. Without it the build dies in
                # _infer_cluster_names_from_matrix on a 512-vs-1024 matmul,
                # after the expensive encoding is already done.
                model, processor = load_clip(model_id=model_id)
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
                    vecs.append(_text2vec(t, modality, model_id=model_id)[None, :])
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
_CLIP = {"model": None, "proc": None, "id": None}
_CLAP = {"model": None, "proc": None}

def _device() -> str:
    # Not torch.cuda.is_available(): that says a driver and card exist, not
    # that this torch build has kernels for them. See arcana/gpu.py.
    return _gpu.device()

def load_clip(device: str | None = None, model_id: str | None = None):
    """
    Load an image encoder, caching one at a time.

    The cache is keyed by model id: indexing two datasets with different
    encoders in one session used to silently reuse whichever loaded first, and
    the second dataset would then be built with the wrong model while recording
    the right one.
    """
    device = device or _device()
    model_id = model_id or CLIP_MODEL_ID
    if _CLIP["model"] is None or _CLIP.get("id") != model_id:
        from transformers import CLIPModel, CLIPProcessor
        m = CLIPModel.from_pretrained(model_id)
        if device == "cuda":
            m = m.to("cuda")
            # Half precision only from Volta up. A pre-Volta card runs .half()
            # slowly and less accurately, and the result goes into an index
            # that is then indistinguishable from an fp32 one.
            if _gpu.use_fp16():
                m = m.half()
        else:
            m = m.to("cpu")
        m.eval()
        p = CLIPProcessor.from_pretrained(model_id)
        _CLIP.update(model=m, proc=p, id=model_id)
    return _CLIP["model"], _CLIP["proc"]

def load_clap(device: str | None = None):
    device = device or _device()
    if _CLAP["model"] is None:
        from transformers import ClapModel, ClapProcessor
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
def img2vec_clip(image_bgr: np.ndarray, model_id: str | None = None) -> np.ndarray:
    """Input: BGR uint8 image (cv2)."""
    model, processor = load_clip(model_id=model_id)
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    px = processor(images=[rgb], return_tensors="pt").pixel_values.to(model.device)
    with torch.no_grad():
        vec = model.get_image_features(px).squeeze().detach().cpu().float().numpy()
    return vec

def txt2vec_clip(text: str, model_id: str | None = None) -> np.ndarray:
    model, processor = load_clip(model_id=model_id)
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
    # torchaudio 2.9 removed its own decoding backends: torchaudio.load now
    # delegates to TorchCodec and raises ImportError if it is absent. That is a
    # hard failure for every audio dataset, so soundfile -- which is already a
    # dependency and reads WAV/FLAC/OGG natively -- is tried first, and
    # torchaudio only as a fallback for what soundfile cannot open (mp3, m4a).
    wav = sr = None
    if sf is not None:
        try:
            wav, sr = sf.read(path, always_2d=False, dtype="float32")
            if getattr(wav, "ndim", 1) == 2:
                wav = wav.mean(axis=1)
        except Exception:
            wav = sr = None

    if wav is None:
        if torchaudio is None:
            raise RuntimeError(
                f"Could not read {os.path.basename(path)}: soundfile failed and "
                "torchaudio is not installed."
            )
        try:
            t_wav, sr = torchaudio.load(path)      # (ch, n)
            wav = t_wav.mean(dim=0).numpy()
        except ImportError as e:
            raise RuntimeError(
                f"Could not read {os.path.basename(path)}: soundfile could not "
                f"open it and torchaudio needs TorchCodec ({e}). Install "
                "torchcodec, or convert the file to WAV or FLAC."
            ) from e

    # resample
    if sr != target_sr:
        if torchaudio is not None:
            wav = torchaudio.functional.resample(
                torch.from_numpy(np.ascontiguousarray(wav)), sr, target_sr).numpy()
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
def build(glob_path: str, index_path: str, batch_size: int = 32, modality: str = "image",
          model_id: str | None = None, progress=None) -> tuple[Index, dict]:
    """
    Build a cosine index for the given media files.

    modality: "image" or "audio"
    model_id: which encoder to use; defaults to the modality's usual one
    progress: optional callable(done, total, message) so a UI can show a bar.
              Indexing runs for minutes to hours, so a caller that cannot report
              progress leaves the user staring at nothing.
    """
    def _report(done, total, message=""):
        if progress is not None:
            progress(done, total, message)

    # discover files
    _report(0, 0, "Scanning for files")
    all_paths = glob(glob_path, recursive=True)
    paths = [p for p in all_paths if (is_image(p) if modality == "image" else is_audio(p))]
    print(f"[INFO] Found {len(paths)} {modality} files to index.")
    if not paths:
        raise SystemExit("No files found for indexing.")
    _report(0, len(paths), f"Found {len(paths):,} files")

    # probe ndim from first vector
    if modality == "image":
        probe = imread_unicode(paths[0])
        if probe is None:
            raise SystemExit(f"Failed to read first image: {paths[0]}")
        v0 = img2vec_clip(probe, model_id=model_id)
    else:
        # PROBE (before creating the index)
        a, sr = read_audio_mono(paths[0], target_sr=48000, seconds=None, pad=False)
        v0 = aud2vec_clap(a, sr)


    ndim = int(v0.shape[-1])
    # usearch defaults to bf16, which silently quantises every embedding (~1e-2
    # absolute error) and makes the index the only, lossy, copy of the vectors.
    # Ask for f32 explicitly so precision is a decision rather than a default.
    index = Index(ndim=ndim, metric="cos", dtype="f32")
    idx2path: dict[int, str] = {}

    # Keys must stay contiguous. They are used downstream as positional row
    # indices into the latent DataFrame, so a gap left by an unreadable file
    # silently shifts every later item onto the wrong point.
    next_key = 0

    if modality == "image":
      # OpenCV threads inside a single decode by default, which fights the pool
      # below for the same cores. Pin it to one for the duration and restore it
      # afterwards, so nothing else in the process is affected.
      _cv_threads = cv2.getNumThreads()
      cv2.setNumThreads(1)
      # One pool for the whole run: a batch is 32 images, so creating a pool per
      # batch would make thread start-up a measurable share of the work.
      with ThreadPoolExecutor(max_workers=_decode_workers()) as _decoder:
        for batch_start in tqdm(range(0, len(paths), batch_size), desc="Indexing images"):
            _report(batch_start, len(paths), "Encoding images")
            batch_paths = paths[batch_start : batch_start + batch_size]
            imgs = []
            ok_paths = []
            # map keeps input order, so imgs and ok_paths stay aligned with each
            # other and with the vectors the encoder returns. A bad file yields
            # None and is dropped from both.
            for p, im in zip(batch_paths, _decoder.map(imread_for_encoder, batch_paths)):
                if im is None:
                    print(f"[WARN] bad image, skipping: {p}")
                    continue
                imgs.append(im)
                ok_paths.append(p)
            if not ok_paths:
                continue
            # Encode batch
            model, processor = load_clip(model_id=model_id)
            px = processor(images=[cv2.cvtColor(x, cv2.COLOR_BGR2RGB) for x in imgs],
                           return_tensors="pt").pixel_values.to(model.device)
            with torch.no_grad():
                vecs = model.get_image_features(px).detach().cpu().float().numpy()
            for i, vec in enumerate(vecs):
                # fp16 on the GPU can overflow to inf/nan on a pathological
                # image. usearch stores it happily, and the point then sits at
                # an undefined place on the map matching nothing, with no error
                # raised anywhere.
                if not np.isfinite(vec).all():
                    print(f"[WARN] non-finite embedding, skipping: {ok_paths[i]}")
                    continue
                index.add(next_key, vec)
                idx2path[next_key] = os.path.abspath(ok_paths[i])
                next_key += 1

    else:  # audio
        # MAIN LOOP
        for _i, p in enumerate(tqdm(paths, desc="Indexing audio")):
            _report(_i, len(paths), "Encoding audio")
            try:
                a, sr = read_audio_mono(p, target_sr=48000, seconds=None, pad=False)
                vec = aud2vec_clap(a, sr)
            except Exception as e:
                print(f"[WARN] failed on {p}: {e}")
                continue
            if not np.isfinite(vec).all():
                print(f"[WARN] non-finite embedding, skipping: {p}")
                continue
            index.add(next_key, vec)
            idx2path[next_key] = os.path.abspath(p)
            next_key += 1

    if modality == "image":
        cv2.setNumThreads(_cv_threads)

    if not idx2path:
        raise SystemExit(f"No {modality} files could be read; nothing to index.")
    skipped = len(paths) - len(idx2path)
    if skipped:
        print(f"[INFO] Indexed {len(idx2path)} of {len(paths)} files ({skipped} unreadable).")

    _paths.ensure_dir(os.path.dirname(os.path.abspath(index_path)))
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


def _decode_workers() -> int:
    """
    How many threads to decode images with.

    Decoding dominated indexing and it was serial: ~143 ms per image on the
    machine this was measured on, against 27.6 ms for a ViT-B/32 forward pass
    on the CPU and 0.6 ms on the GPU. That one loop is why a GPU was worth only
    1.2x end-to-end for the default encoder -- the card sat idle for roughly
    94% of a run, waiting for JPEGs.

    Threads, not processes: cv2.imdecode releases the GIL for the whole decode,
    and a process pool inside a frozen app re-launches the application once per
    worker (see _feature_executor). Capped because past a point this is bound by
    the disk, and every in-flight image is a decoded bitmap held in memory.

    Measured on 120 real photographs with a warm page cache, decoding one image:

        serial, OpenCV threading on     79.1 ms
        16 threads, OpenCV threading on 17.8 ms
        22 threads, OpenCV pinned to 1  14.1 ms   <- 5.6x

    OpenCV parallelises inside a single decode by default, so an outer pool
    ends up fighting it for the same cores. Pinning cv2 to one thread while the
    pool is open is worth another ~25%.
    """
    n = os.cpu_count() or 4
    return int(max(2, min(24, n)))


def _feature_executor(n_workers: int):
    """
    Pick a pool for CPU-bound feature extraction.

    Processes normally: palette and style are CPU-bound and a process pool
    sidesteps the GIL entirely.

    Threads when frozen. On Windows multiprocessing spawns rather than forks,
    which re-launches the executable for each worker -- and in a PyInstaller
    build the executable is the whole application. Even with freeze_support()
    in place (without it the workers start a second Dash server and get killed,
    which is exactly what happened: every image failed with "A process in the
    process pool was terminated abruptly" at ~36 minutes each), every worker
    would still pay a full interpreter start plus a torch import before doing
    any work. Palette and style are mostly OpenCV and scikit-learn, both of
    which drop the GIL inside their hot loops, so threads keep most of the
    parallelism at none of that cost.
    """
    if getattr(sys, "frozen", False):
        return ThreadPoolExecutor(max_workers=n_workers), "threads"
    return ProcessPoolExecutor(max_workers=n_workers), "processes"


def extract_additional_features(
    idx2path: dict[int, str],
    name: str,
    features: list[str],
    include_gram: bool = True,
    compact_gram: bool = True,
    gram_pca_dims: int = 0,
    n_workers: int = 1,
    progress=None,
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
                _pool, _kind = _feature_executor(n_workers)
                print(f"[INFO] using {n_workers} {_kind}")
                with _pool as executor:
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
            
            # Batched, not one image at a time. This loop used to call
            # extract_all_style_features per file, which runs VGG19 on a single
            # image: 9,359 photographs took about ninety minutes at ~1.7/s with
            # the device idle between them. CLIP indexing was batched from the
            # start; this was not.
            try:
                from .style import (extract_gram_features_batch,
                                     extract_edge_histogram, extract_texture_lbp)
            except ImportError:
                from style import (extract_gram_features_batch,
                                   extract_edge_histogram, extract_texture_lbp)

            _style_batch = 16
            _pool, _kind = _feature_executor(n_workers)
            print(f"[INFO] using {n_workers} {_kind}, batches of {_style_batch}")
            with _pool as _sx:
                with tqdm(total=len(paths), desc="Style features") as _bar:
                    for _b0 in range(0, len(paths), _style_batch):
                        chunk_ids = sorted_ids[_b0:_b0 + _style_batch]
                        chunk_paths = paths[_b0:_b0 + _style_batch]

                        # Decode in parallel; the VGG pass is the serial part.
                        imgs = list(_sx.map(imread_for_encoder, chunk_paths))
                        keep = [(i, p, im) for i, p, im in
                                zip(chunk_ids, chunk_paths, imgs) if im is not None]
                        for i, p, im in zip(chunk_ids, chunk_paths, imgs):
                            if im is None:
                                print(f"[WARN] Style: could not read {p}; skipping")

                        if not keep:
                            _bar.update(len(chunk_paths))
                            continue

                        k_ids = [k[0] for k in keep]
                        k_imgs = [k[2] for k in keep]

                        # The cheap per-image parts stay parallel.
                        try:
                            edges = list(_sx.map(extract_edge_histogram, k_imgs))
                            lbps = list(_sx.map(extract_texture_lbp, k_imgs))
                        except Exception as e:
                            print(f"[WARN] Style features failed on a batch: {e}")
                            _bar.update(len(chunk_paths))
                            continue

                        grams = None
                        if include_gram:
                            try:
                                grams = extract_gram_features_batch(
                                    k_imgs, compact=compact_gram)
                            except Exception as e:
                                print(f"[WARN] Gram failed on a batch: {e}")
                                _bar.update(len(chunk_paths))
                                continue

                        # Every list here is joined by position to valid_ids, so
                        # an item that cannot supply all of them is skipped
                        # entirely -- appending some but not others silently
                        # shifts every later row onto the wrong image.
                        for j, idx in enumerate(k_ids):
                            if include_gram and (grams is None or grams[j] is None):
                                continue
                            edge_histograms.append(edges[j])
                            lbp_textures.append(lbps[j])
                            if include_gram:
                                gram_features.append(grams[j])
                            valid_ids.append(idx)

                        _bar.update(len(chunk_paths))
                        if progress is not None:
                            progress(_b0 + len(chunk_paths), len(paths),
                                     "Extracting style features")
            
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
        List of (path, similarity) tuples, sorted by relevance (higher = better)
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
        # Euclidean distance - convert to similarity
        query_mom = query_feats['moments']
        distances = []
        for i, db_mom in enumerate(features['moments']):
            dist = float(np.linalg.norm(query_mom - db_mom))
            distances.append((idx2path[ids[i]], dist))
        
        # Convert distances to similarities: sim = 1 / (1 + dist)
        max_dist = max(d[1] for d in distances) if distances else 1.0
        for path, dist in distances:
            sim = 1.0 - (dist / (max_dist + 1e-8))  # Normalize to 0-1, higher = better
            results.append((path, sim))
        results.sort(key=lambda x: -x[1])  # Higher = better
        
    elif method == "emd":
        # Earth Mover's Distance - convert to similarity
        query_dom = query_feats['dominant']
        distances = []
        for i, db_dom in enumerate(features['dominant']):
            dist = emd_palette_distance(query_dom, db_dom, n_colors=n_colors)
            distances.append((idx2path[ids[i]], float(dist)))
        
        # Convert distances to similarities
        max_dist = max(d[1] for d in distances) if distances else 1.0
        for path, dist in distances:
            sim = 1.0 - (dist / (max_dist + 1e-8))  # Normalize to 0-1, higher = better
            results.append((path, sim))
        results.sort(key=lambda x: -x[1])  # Higher = better
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

        # Stored Gram vectors were compressed with PCA at index time (db.py fits a
        # PCA and saves its basis). The query's raw Gram is full-width, so it must
        # go through the same basis before any comparison -- without this the
        # matmul is a shape error and gram search never works at all.
        if 'gram_pca_components' in features and query_vec.shape[1] != db_vecs.shape[1]:
            comps = features['gram_pca_components']          # (n_out, n_in)
            mean = features.get('gram_pca_mean')
            if query_vec.shape[1] != comps.shape[1]:
                raise ValueError(
                    f"Query Gram is {query_vec.shape[1]}-d but this index's PCA expects "
                    f"{comps.shape[1]}-d. The index was built with a different Gram "
                    f"setting (--full_gram vs the default compact); rebuild it or query "
                    f"with matching settings."
                )
            if mean is not None:
                query_vec = query_vec - mean.reshape(1, -1)
            query_vec = query_vec @ comps.T                  # -> (1, n_out)

        if query_vec.shape[1] != db_vecs.shape[1]:
            raise ValueError(
                f"Gram dimensionality mismatch: query is {query_vec.shape[1]}-d, index is "
                f"{db_vecs.shape[1]}-d, and the index stores no PCA basis to reconcile them."
            )

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
        # The floor scales with the collection. Without it the silhouette sweep
        # starts at k=2 and, on CLIP embeddings, essentially always stops
        # there -- see auto_k for why that produces two meaningless names.
        # The metric still chooses within the remaining range, so a collection
        # that genuinely wants more clusters can still ask for them.
        sized = auto_k(len(X_for_kmeans))
        k_min = max(int(os.getenv("ARCANA_K_MIN", "2")), sized)
        k_max = max(int(os.getenv("ARCANA_K_MAX", "20")), k_min)
        metric = os.getenv("ARCANA_K_METRIC", "silhouette")
        # k must stay below the number of points or KMeans cannot fit.
        k_max = min(k_max, max(2, len(X_for_kmeans) - 1))
        k_min = min(k_min, k_max)
        best_k, scores = choose_k(X_for_kmeans, k_min=k_min, k_max=k_max, metric=metric)
        print(f"[auto-k] {len(X_for_kmeans)} items -> floor {sized}; "
              f"selected k={best_k} via {metric} in [{k_min},{k_max}]")
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
    parser.add_argument("--model", type=str, default=None,
        help="Encoder to index with. Defaults to the best one this "
             "machine can run comfortably; see arcana/models.py.")
    parser.add_argument("--thumbnails", action="store_true",
        help="Embed 192px previews in the portable bundle, so it can be browsed "
             "without the original files.")
    parser.add_argument("--workers", type=int, default=0,
        help="Number of parallel workers for palette/style extraction (0=auto, uses CPU count).")

    return parser.parse_args()


# --------------------------------------------------------------------------------------
# Portable bundle output
# --------------------------------------------------------------------------------------
def write_bundle(name: str, modality: str, media_root: str, index, idx2path: dict,
                 coords, cluster_ids, labels, model_id: str,
                 feature_paths: dict | None = None, n_components: int = 2,
                 thumbnails: bool = False) -> str:
    """
    Write a self-describing .arcana bundle beside the legacy pickles.

    The legacy index stores absolute paths, so a dataset dies the moment its
    folder moves. A bundle stores a content fingerprint per item plus a path
    relative to `media_root`, so relocating is a rescan (see arcana-relocate)
    rather than a rebuild.
    """
    try:
        from .bundle import Bundle, BundleWriter, Item, ModelSpec, SUFFIX
    except ImportError:
        from bundle import Bundle, BundleWriter, Item, ModelSpec, SUFFIX

    keys = sorted(int(k) for k in idx2path.keys())
    vecs = np.asarray([index.get(k) for k in keys], dtype=np.float32)
    if vecs.ndim == 3 and vecs.shape[1] == 1:
        vecs = vecs[:, 0, :]

    root = os.path.abspath(media_root)
    if not os.path.isdir(root):
        root = os.path.dirname(root)

    items, keep = [], []
    for row, k in enumerate(keys):
        src = str(idx2path[k])
        try:
            it = Item.for_file(src, root)
        except OSError:
            print(f"[WARN] bundle: could not fingerprint {src}; skipping")
            continue
        it.cluster_id = int(cluster_ids[row]) if cluster_ids is not None else -1
        it.label = str(labels[row]) if labels is not None else ""
        items.append(it)
        keep.append(row)

    if not items:
        raise RuntimeError("no items could be fingerprinted")

    seen, uniq_items, uniq_keep = set(), [], []
    for it, row in zip(items, keep):
        if it.id in seen:
            continue
        seen.add(it.id)
        uniq_items.append(it)
        uniq_keep.append(row)
    if len(uniq_items) != len(items):
        print(f"[INFO] bundle: collapsed {len(items) - len(uniq_items)} byte-identical duplicate(s)")
    items, keep = uniq_items, uniq_keep

    vecs = vecs[keep]
    lay = np.asarray(coords, dtype=np.float32)[keep] if coords is not None else None

    out_dir = _paths.ensure_dir(_paths.subdir("bundles"))
    out_path = os.path.join(out_dir, f"{name}_{modality}{SUFFIX}")

    with BundleWriter(out_path, name=name,
                      model=ModelSpec(id=model_id, dim=int(vecs.shape[1]), modality=modality),
                      root=root, source=f"built by arcana-build-latent from {media_root}",
                      tool_version="arcana.db/1") as w:
        w.set_items(items)
        w.set_vectors(vecs, precision="f32")
        if lay is not None:
            w.set_layout(lay, algo="tsne", params={"n_components": int(n_components)})
        for block, path in (feature_paths or {}).items():
            if block not in ("palette", "style") or not os.path.exists(path):
                continue
            with np.load(path) as z:
                arrays = {k: z[k] for k in z.files}
            ids = arrays.get("ids")
            if ids is not None:
                key_to_row = {keys[old]: new for new, old in enumerate(keep)}
                rows, sel = [], []
                for pos, oid in enumerate(np.asarray(ids).astype(int)):
                    nr = key_to_row.get(int(oid))
                    if nr is not None:
                        rows.append(nr)
                        sel.append(pos)
                if rows:
                    sel_arr = np.asarray(sel, dtype=int)
                    remapped = {"ids": np.asarray(rows, dtype=np.int32)}
                    for kk, arr in arrays.items():
                        if kk == "ids":
                            continue
                        remapped[kk] = arr[sel_arr] if arr.shape[:1] == np.asarray(ids).shape else arr
                    arrays = remapped
            w.add_feature_block(block, arrays)
        if thumbnails and modality == "image":
            made = 0
            for it in items:
                src = os.path.join(root, *it.rel_path.split("/"))
                data = _thumb_bytes(src)
                if data:
                    w.add_thumbnail(it.id, data)
                    made += 1
            print(f"[INFO] bundle: embedded {made} thumbnails")

    return out_path


def _thumb_bytes(path: str, max_side: int = 192) -> bytes | None:
    try:
        from PIL import Image
        import io
        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((max_side, max_side), Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, "WEBP", quality=80, method=4)
            return buf.getvalue()
    except Exception:
        return None



# --------------------------------------------------------------------------------------
# Label caches, one per encoder
# --------------------------------------------------------------------------------------
def label_cache_status(modality: str = "image", labels_src: str | None = None) -> list[dict]:
    """
    For each encoder Arcana offers, is its label matrix already built?

    Cluster names come from comparing label embeddings against image centroids,
    so the labels must be encoded by the SAME model as the images. Every model
    therefore needs its own cache; a missing one is not an error, just work that
    has to happen before that model can name anything.
    """
    try:
        from . import models as _models
    except ImportError:
        import models as _models

    src = labels_src or DEFAULT_LABELS.get(modality)
    labels, cache_base = _read_label_list(src)
    out = []
    for m in _models.for_modality(modality):
        model_tag = hashlib.md5(m.id.encode("utf-8")).hexdigest()[:8]
        cache_key = f"{modality}|{m.id}|{cache_base}"
        disk_hash = hashlib.md5(cache_key.encode("utf-8")).hexdigest()[:12]
        path = os.path.join(db_dir, f"label_cache_{modality}_{disk_hash}_{model_tag}.pkl")
        out.append({
            "model_id": m.id, "label": m.label, "dim": m.dim,
            "n_labels": len(labels), "path": path,
            "ready": os.path.exists(path),
            "model_downloaded": _models.is_downloaded(m.id),
        })
    return out


def warm_label_cache(model_id: str, modality: str = "image",
                     labels_src: str | None = None, progress=None) -> dict:
    """
    Build (and persist) the label matrix for one encoder.

    Cheap -- a hundred short strings -- but it must happen before that encoder
    can name clusters, and doing it up front means an index run does not stop
    halfway to encode labels.
    """
    src = labels_src or DEFAULT_LABELS.get(modality)
    labels, cache_base = _read_label_list(src)
    if not labels:
        return {"model_id": model_id, "n_labels": 0, "dim": 0, "skipped": "no labels"}
    if progress is not None:
        progress(0.0, f"Encoding {len(labels)} labels", 0, len(labels))
    _paths.ensure_dir(db_dir)
    texts, M = _encode_label_matrix(labels, modality, cache_base, model_id=model_id)
    if progress is not None:
        progress(1.0, f"{len(texts)} labels ready", len(texts), len(texts))
    return {"model_id": model_id, "n_labels": len(texts),
            "dim": int(M.shape[1]) if M.size else 0}


def warm_all_label_caches(modality: str = "image", only_downloaded: bool = True,
                          labels_src: str | None = None, progress=None) -> list[dict]:
    """
    Make every encoder's label matrix ready.

    only_downloaded=True skips models whose weights are not local, so this never
    silently pulls gigabytes; pass False to accept the download.
    """
    status = label_cache_status(modality, labels_src)
    todo = [s for s in status
            if not s["ready"] and (s["model_downloaded"] or not only_downloaded)]
    done = []
    for i, s in enumerate(todo):
        if progress is not None:
            progress(i / max(1, len(todo)), f"Preparing labels for {s['label']}",
                     i, len(todo))
        try:
            done.append(warm_label_cache(s["model_id"], modality, labels_src))
        except Exception as e:
            done.append({"model_id": s["model_id"], "error": f"{type(e).__name__}: {e}"})
    if progress is not None:
        progress(1.0, "Labels ready", len(todo), len(todo))
    return done


def _scaled(progress, lo: float, hi: float):
    """Adapt build()'s (done, total, message) into a slice of an overall bar."""
    if progress is None:
        return None

    def cb(done, total, message=""):
        frac = lo + (hi - lo) * ((done / total) if total else 0.0)
        progress(frac, message, done, total)
    return cb


def index_dataset(
    media_path: str,
    name: str,
    *,
    modality: str = "image",
    n_components: int = 2,
    model_id: str | None = None,
    features: str = "clip",
    labels: str | None = None,
    k: int = 0,
    k_min: int = 2,
    k_max: int = 20,
    k_metric: str = "silhouette",
    no_gram: bool = False,
    full_gram: bool = False,
    gram_pca: int = 512,
    reuse_index: bool = False,
    workers: int = 0,
    thumbnails: bool = False,
    progress=None,
    should_cancel=None,
) -> dict:
    """
    Build a dataset end to end: encode, extract features, lay out, save.

    The single implementation behind both `arcana-build-latent` and the app's
    dataset manager. main() is a thin wrapper over this, so the CLI cannot drift
    from what the GUI runs -- they used to be the same code only by accident,
    and a bug living solely in main() shipped unnoticed.

    progress: callable(fraction, message, done, total)
    should_cancel: callable() -> bool, checked between phases
    """
    def report(frac, message="", done=0, total=0):
        if progress is not None:
            progress(frac, message, done, total)

    def check_cancel():
        if should_cancel is not None and should_cancel():
            raise KeyboardInterrupt("cancelled")

    if model_id is None:
        model_id = CLIP_MODEL_ID if modality == "image" else CLAP_MODEL_ID

    def _to_glob(p: str) -> str:
        if any(ch in p for ch in "*?[]"):
            return p
        p_abs = os.path.abspath(p)
        return os.path.join(p_abs, "**", "*") if os.path.isdir(p_abs) else p_abs

    glob_arg = _to_glob(media_path)
    index_name = os.path.join(db_dir, f"index_{name}_{modality}.pkl")
    latent_name = os.path.join(
        latents_dir, f"latent_space_{name}_{modality}_{n_components}d.pkl")
    _paths.ensure_dir(db_dir)
    _paths.ensure_dir(latents_dir)

    print("path to index:       ", index_name)
    print("path to latent space:", latent_name)
    print("search path:         ", glob_arg)
    print("modality:            ", modality)
    print("model:               ", model_id)

    # ---- labels -------------------------------------------------------------
    report(0.01, "Preparing labels")
    label_src = labels
    if not label_src:
        cand = DEFAULT_LABELS.get(modality)
        if cand and os.path.exists(cand):
            print(f"[INFO] Using default labels: {cand}")
            label_src = cand
    label_texts, cache_base = _read_label_list(label_src)
    if label_texts:
        label_texts, label_mat = _encode_label_matrix(
            label_texts, modality, cache_base, model_id=model_id)
    else:
        label_mat = np.zeros((0, 1), dtype=np.float32)
    check_cancel()

    # ---- encode -------------------------------------------------------------
    if reuse_index and os.path.exists(index_name):
        report(0.05, "Reusing the existing index")
        print(f"[INFO] Reusing existing index: {index_name}")
        with open(index_name, "rb") as fh:
            saved_index, idx2path = pickle.load(fh)
        index = Index.restore(saved_index)
        print(f"[INFO] Loaded {len(idx2path)} indexed paths.")
    else:
        index, idx2path = build(glob_arg, index_name, modality=modality,
                                model_id=model_id,
                                progress=_scaled(progress, 0.03, 0.65))
    check_cancel()

    # ---- palette / style ----------------------------------------------------
    feature_paths: dict = {}
    feature_list = [x.strip().lower() for x in features.split(",")]
    if modality == "image" and any(x in feature_list for x in ["palette", "style"]):
        report(0.66, "Extracting palette and style features")
        additional = [x for x in feature_list if x in ("palette", "style")]
        n_workers = workers if workers > 0 else multiprocessing.cpu_count()
        feature_paths = extract_additional_features(
            idx2path=idx2path, name=name, features=additional,
            include_gram=(not no_gram), compact_gram=(not full_gram),
            gram_pca_dims=gram_pca, n_workers=n_workers,
            # Palette and style together are the longest phase on a large
            # library -- about ninety minutes for 9,359 images -- and the job
            # card used to sit at one position for all of it with no counter,
            # which is indistinguishable from being stuck.
            progress=lambda d, t, m: report(
                0.66 + 0.24 * (d / t if t else 0), m, done=d, total=t),
        )
        for ftype, fpath in feature_paths.items():
            print(f"  {ftype}: {fpath}")
    check_cancel()

    # ---- layout -------------------------------------------------------------
    report(0.80, "Laying out the latent space")
    if k <= 0:
        os.environ["ARCANA_K_MIN"] = str(k_min)
        os.environ["ARCANA_K_MAX"] = str(k_max)
        os.environ["ARCANA_K_METRIC"] = k_metric

    coords, paths, cluster_ids, inferred_names = latent_space(
        index=index, idx2path=idx2path, n_components=n_components,
        modality=modality, n_clusters=(0 if k <= 0 else int(k)),
        label_texts=label_texts, label_mat_norm=label_mat,
    )
    check_cancel()

    # ---- save ---------------------------------------------------------------
    report(0.92, "Saving")
    cols = ["x", "y"] if n_components == 2 else ["x", "y", "z"]
    df = pd.DataFrame(coords, columns=cols)
    df["path"] = paths
    df["cluster_id"] = cluster_ids.astype(int)
    df["label"] = [nm if nm else f"C{int(cid)}"
                   for nm, cid in zip(inferred_names, cluster_ids)]
    df.to_pickle(latent_name)
    print(f"[OK] Saved latent DataFrame to {latent_name}")

    result = {"name": name, "modality": modality, "n_items": len(idx2path),
              "model_id": model_id, "index": index_name, "latent": latent_name,
              "features": sorted(feature_paths), "bundle": None}

    report(0.95, "Writing the portable bundle")
    try:
        result["bundle"] = write_bundle(
            name=name, modality=modality, media_root=media_path,
            index=index, idx2path=idx2path, coords=coords,
            cluster_ids=cluster_ids, labels=df["label"].tolist(),
            model_id=model_id, feature_paths=feature_paths,
            n_components=n_components, thumbnails=thumbnails,
        )
        print(f"[OK] Saved portable bundle to {result['bundle']}")
    except (NameError, AttributeError, TypeError, KeyError, IndexError):
        # A bug in our own code. Do not downgrade it to a warning -- that is
        # exactly how an unbound feature_paths went unnoticed and silently
        # skipped the bundle on every default build.
        raise
    except Exception as e:
        print(f"[WARN] Could not write the portable bundle: {type(e).__name__}: {e}")
        print("       The legacy .pkl files were written and remain usable.")

    report(1.0, f"Indexed {len(idx2path):,} items")
    return result


def main():
    args = parse_args()
    index_dataset(
        args.imgs_path, args.name,
        modality=args.modality, n_components=args.n_components,
        model_id=getattr(args, "model", None),
        features=args.features, labels=args.labels,
        k=args.k, k_min=args.k_min, k_max=args.k_max, k_metric=args.k_metric,
        no_gram=args.no_gram, full_gram=args.full_gram, gram_pca=args.gram_pca,
        reuse_index=args.reuse_index, workers=args.workers,
        thumbnails=args.thumbnails,
    )


if __name__ == "__main__":
    main()

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

import numpy as np
import torch
from tqdm import tqdm
import pandas as pd

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

from usearch.index import Index
from transformers import CLIPModel, CLIPProcessor, ClapModel, ClapProcessor

# Optional audio backends
try:
    import torchaudio
except Exception:  # pragma: no cover
    torchaudio = None
try:
    import soundfile as sf
except Exception:  # pragma: no cover
    sf = None

torch.set_grad_enabled(False)

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
script_root = os.path.dirname(os.path.abspath(__file__))
IMAGES_ROOT = os.path.abspath(os.path.join(script_root, "..", "images"))

db_dir = os.path.join(script_root, "databases")
latents_dir = os.path.join(script_root, "latents")
os.makedirs(db_dir, exist_ok=True)
os.makedirs(latents_dir, exist_ok=True)

# --------------------------------------------------------------------------------------
# File types
# --------------------------------------------------------------------------------------
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}

def is_image(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in IMAGE_EXTS

def is_audio(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in AUDIO_EXTS

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
        m = CLIPModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        if device == "cuda":
            m = m.to("cuda").half()
        else:
            m = m.to("cpu")
        m.eval()
        p = CLIPProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        _CLIP.update(model=m, proc=p)
    return _CLIP["model"], _CLIP["proc"]

def load_clap(device: str | None = None):
    device = device or _device()
    if _CLAP["model"] is None:
        m = ClapModel.from_pretrained("laion/clap-htsat-fused")
        # Keep CLAP in FP32 (BN layers are happier; avoids dtype mismatch)
        if device == "cuda":
            m = m.to("cuda")   # <-- no .half()
        else:
            m = m.to("cpu")
        m.eval()
        p = ClapProcessor.from_pretrained("laion/clap-htsat-fused")
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
# Latent space (TSNE + kmeans labels for coloring)
# --------------------------------------------------------------------------------------
def latent_space(index: Index, idx2path: dict[int, str], n_components: int = 2):
    vecs = []
    paths = []
    for kid in tqdm(idx2path.keys(), desc="Collecting vectors"):
        vecs.append(index.get(kid))
        paths.append(os.path.abspath(idx2path[kid]))
    vecs = np.asarray(vecs, dtype=np.float32)

    # t-SNE (2D/3D)
    perplexity = int(np.clip(math.ceil(len(vecs) / 10), 5, 30))
    if n_components == 2:
        tsne = TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto")
        coords = tsne.fit_transform(vecs)
    elif n_components == 3:
        tsne = TSNE(n_components=3, perplexity=perplexity, init="pca", learning_rate="auto")
        coords = tsne.fit_transform(vecs)
    else:
        raise ValueError("n_components must be 2 or 3")

    # simple clustering for color labels
    try:
        kmeans = KMeans(n_clusters=10, random_state=0, n_init="auto").fit(coords)
        labels = kmeans.labels_
    except Exception:
        labels = np.zeros(len(coords), dtype=int)

    return coords, paths, labels

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

    index, idx2path = build(glob_arg, index_name, modality=args.modality)
    coords, paths, labels = latent_space(index, idx2path, n_components=args.n_components)

    if args.n_components == 2:
        df = pd.DataFrame(coords, columns=["x", "y"])
    else:
        df = pd.DataFrame(coords, columns=["x", "y", "z"])
    df["path"] = paths
    df["label"] = labels.astype(int)

    df.to_pickle(latent_name)
    print(f"[OK] Saved latent DataFrame to {latent_name}")

if __name__ == "__main__":
    main()

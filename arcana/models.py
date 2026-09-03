# models.py — the encoders Arcana can index with, and what each one costs
#
# The model id used to be hardcoded in two places (db.py and arcana.py), which
# meant a dataset could not record what built it and the app could not offer a
# choice. Everything about an encoder now lives here.
#
# The timings are measured, not guessed: 24 MP JPEGs on a 22-core CPU and an
# RTX 4090 Laptop, with JPEG decode timed separately from the forward pass. See
# docs/hardening-audit.md. They exist so the UI can tell someone "about 20
# minutes" instead of showing an unbounded spinner.

from __future__ import annotations

import os
from dataclasses import dataclass, asdict

try:
    from . import gpu as _gpu
except ImportError:
    import gpu as _gpu

IMAGE = "image"
AUDIO = "audio"


@dataclass(frozen=True)
class ModelInfo:
    id: str                  # HuggingFace repo id
    label: str               # what a person sees
    modality: str
    dim: int                 # embedding dimension
    download_mb: int         # measured size of the weights actually fetched
    cpu_ms: float            # measured ms per image, model forward only
    gpu_ms: float            # same, on the reference GPU
    quality: str             # relative retrieval quality, plain words
    blurb: str

    def as_dict(self) -> dict:
        return asdict(self)


# Ordered cheapest first: that is also the order a chooser should present.
MODELS: tuple[ModelInfo, ...] = (
    ModelInfo(
        id="laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
        label="CLIP ViT-B/32 — fast",
        modality=IMAGE, dim=512, download_mb=605,
        cpu_ms=21.9, gpu_ms=0.2,
        quality="good",
        blurb="Indexes about 10,000 photos in seven minutes without a graphics card. "
              "The right choice unless you have a GPU or a small library.",
    ),
    ModelInfo(
        id="laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
        label="CLIP ViT-L/14 — balanced",
        modality=IMAGE, dim=768, download_mb=1711,
        cpu_ms=378.5, gpu_ms=3.3,
        quality="better",
        blurb="Noticeably better at specific prompts. Comfortable with a GPU; "
              "over an hour per 10,000 photos without one.",
    ),
    ModelInfo(
        id="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        label="CLIP ViT-H/14 — best quality",
        modality=IMAGE, dim=1024, download_mb=3945,
        cpu_ms=715.7, gpu_ms=7.1,
        quality="best",
        blurb="The strongest retrieval, and what existing Arcana datasets were "
              "built with. Needs a GPU to be practical.",
    ),
    ModelInfo(
        id="laion/clap-htsat-fused",
        label="CLAP — audio",
        modality=AUDIO, dim=512, download_mb=614,
        cpu_ms=120.0, gpu_ms=4.0,
        quality="good",
        blurb="The only audio encoder Arcana ships with.",
    ),
)

BY_ID = {m.id: m for m in MODELS}

# What db.py and arcana.py used before this module existed. Datasets built then
# carry no model id, so this is what we assume they used.
LEGACY_IMAGE_ID = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
LEGACY_AUDIO_ID = "laion/clap-htsat-fused"


def get(model_id: str) -> ModelInfo | None:
    return BY_ID.get(model_id)


def for_modality(modality: str) -> list[ModelInfo]:
    return [m for m in MODELS if m.modality == modality]


def default_for(modality: str, has_gpu: bool | None = None) -> ModelInfo:
    """
    The model to preselect.

    Without a GPU the largest model is a multi-hour job on an ordinary library,
    so the fast one is the honest default; with one, the model is nearly free
    and quality should win.
    """
    if has_gpu is None:
        has_gpu = gpu_available()
    if modality == AUDIO:
        return BY_ID[LEGACY_AUDIO_ID]
    return BY_ID[LEGACY_IMAGE_ID if has_gpu else "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"]


def gpu_available() -> bool:
    """
    True only if a GPU is present AND this torch build can run on it.

    This used to be bool(torch.cuda.is_available()), which is the question
    "is there a card?" rather than "will anything work?". It feeds
    default_for(), so on a card with no compiled kernels it promoted the user
    to ViT-H/14 and a 3,945 MB download before anything failed.
    """
    return _gpu.available()


# --------------------------------------------------------------------------------------
# estimates
# --------------------------------------------------------------------------------------
# Everything an image costs before the model sees it, measured per image on 24 MP
# JPEGs with the decode pool running:
#
#     decode, 1/4 scale, parallel     4.5 ms   <- DECODE_MS_DEFAULT
#     cvtColor + CLIP image processor 13.6 ms  <- PREPROCESS_MS
#
# This used to be a single 143 ms constant, because every photograph was decoded
# at full resolution and then resized from 24 megapixels down to 224x224 -- a
# step that alone was 83% of indexing. cvio.imread_for_encoder decodes at
# reduced scale now. Leaving 143 here would have made every estimate in the
# panel about eight times too pessimistic for a GPU run.
#
# They are separate constants because measure_decode_ms() can measure the first
# against the user's actual folder -- a 2 MP PNG and a 24 MP JPEG differ by an
# order of magnitude -- while the second is fixed by the encoder's input size
# and does not vary with the original.
DECODE_MS_DEFAULT = 4.5
PREPROCESS_MS = 13.6


def estimate_seconds(model: ModelInfo, n_items: int, has_gpu: bool | None = None,
                     cores: int | None = None, decode_ms: float | None = None) -> float:
    """
    Seconds to index n_items.

    decode_ms: measured cost of reading and decoding one file from the actual
    folder. Pass it when you can -- decode dominates on a GPU, and it varies by
    an order of magnitude between a 2 MP PNG and a 24 MP raw-ish JPEG, so a
    fixed constant is a guess where a measurement is cheap.
    """
    if has_gpu is None:
        has_gpu = gpu_available()
    decode = DECODE_MS_DEFAULT if decode_ms is None else decode_ms
    model_ms = model.gpu_ms if has_gpu else model.cpu_ms
    return ((decode + PREPROCESS_MS + model_ms) * n_items) / 1000.0


def humanize(seconds: float) -> str:
    if seconds < 90:
        return f"{max(1, round(seconds))} seconds"
    minutes = seconds / 60
    if minutes < 90:
        return f"about {round(minutes)} minutes"
    hours = minutes / 60
    if hours < 24:
        return f"about {hours:.1f} hours"
    return f"about {hours / 24:.1f} days"


def estimate_text(model: ModelInfo, n_items: int, has_gpu: bool | None = None,
                  decode_ms: float | None = None) -> str:
    if not n_items:
        return ""
    return humanize(estimate_seconds(model, n_items, has_gpu, decode_ms=decode_ms))


def measure_decode_ms(paths: list[str], sample: int = 6) -> float | None:
    """
    Time decoding a few real files so an estimate reflects this folder.

    Times imread_for_encoder, which is what build() calls -- timing a
    full-resolution imread_unicode would report roughly four times the cost
    that indexing will actually pay, and the panel would quote a number nobody
    could reproduce.

    The result covers decode only; estimate_seconds adds the fixed
    colour-conversion and processor cost on top.

    Returns None if nothing could be read; callers then fall back to the
    default constant.
    """
    import random
    import time
    if not paths:
        return None
    try:
        from .cvio import imread_for_encoder
    except ImportError:
        try:
            from cvio import imread_for_encoder
        except ImportError:
            return None
    picks = random.Random(0).sample(paths, min(sample, len(paths)))
    times = []
    for p in picks:
        t0 = time.perf_counter()
        try:
            img = imread_for_encoder(p)
        except Exception:
            continue
        if img is None:
            continue
        times.append((time.perf_counter() - t0) * 1000.0)
    if not times:
        return None
    times.sort()
    return times[len(times) // 2]           # median, so one slow read cannot skew it


# --------------------------------------------------------------------------------------
# is it already downloaded?
# --------------------------------------------------------------------------------------
def is_downloaded(model_id: str) -> bool:
    """
    True when the weights are in the local HuggingFace cache.

    Checked without importing transformers -- this runs while building UI, and
    importing transformers costs seconds.
    """
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    for fname in ("model.safetensors", "pytorch_model.bin"):
        try:
            hit = try_to_load_from_cache(model_id, fname)
        except Exception:
            continue
        if isinstance(hit, str) and os.path.exists(hit):
            return True
    return False


def cache_dir() -> str:
    return (os.environ.get("HF_HOME")
            or os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))


def catalogue(modality: str = IMAGE, has_gpu: bool | None = None,
              n_items: int = 0, decode_ms: float | None = None) -> list[dict]:
    """Everything a chooser needs, in one call."""
    if has_gpu is None:
        has_gpu = gpu_available()
    out = []
    for m in for_modality(modality):
        d = m.as_dict()
        d["downloaded"] = is_downloaded(m.id)
        d["estimate"] = estimate_text(m, n_items, has_gpu, decode_ms) if n_items else ""
        out.append(d)
    return out


# --------------------------------------------------------------------------------------
# downloading, with progress
# --------------------------------------------------------------------------------------
# The redundant weight formats are why a cached ViT-H directory is 5.3 GB when
# the weights transformers actually loads are 3.9 GB. Skipping them is most of
# the difference between a tolerable first run and a bad one.
_IGNORE = [
    "*.msgpack", "*.h5", "*.ot",            # flax / tensorflow copies
    "open_clip_*",                          # open_clip's own duplicate weights
    "*.onnx", "*.onnx_data",
    "*/coreml/*", "*.mlpackage/*",
]


def download_model(model_id: str, handle=None, *, ignore_extra_formats: bool = True) -> str:
    """
    Fetch an encoder into the local cache, reporting progress to a job handle.

    huggingface_hub drives its own tqdm bars; we hand it a tqdm subclass that
    forwards totals into the job instead of writing to a console that a
    packaged app does not have.
    """
    from huggingface_hub import snapshot_download
    from tqdm.auto import tqdm as _tqdm

    info = get(model_id)
    label = info.label if info else model_id
    expected = (info.download_mb if info else 0) * 1e6

    state = {"seen": 0.0}

    class _JobTqdm(_tqdm):
        def __init__(self, *a, **kw):
            kw.setdefault("disable", False)
            super().__init__(*a, **kw)

        def update(self, n=1):
            super().update(n)
            if handle is None:
                return
            state["seen"] += (n or 0)
            total = expected or (self.total or 0)
            got_mb = state["seen"] / 1e6
            if total:
                handle.update(
                    fraction=min(0.999, state["seen"] / total),
                    message=f"Downloading {label}",
                    detail=f"{got_mb:,.0f} MB of about {total / 1e6:,.0f} MB",
                )
            else:
                handle.update(message=f"Downloading {label}",
                              detail=f"{got_mb:,.0f} MB")

    if handle is not None:
        handle.update(fraction=0.0, message=f"Downloading {label}",
                      detail=f"about {expected / 1e6:,.0f} MB")

    path = snapshot_download(
        model_id,
        ignore_patterns=_IGNORE if ignore_extra_formats else None,
        tqdm_class=_JobTqdm,
        max_workers=4,
    )
    if handle is not None:
        handle.update(fraction=1.0, message=f"{label} ready", detail="")
    return path


def ensure_model(model_id: str, handle=None) -> str | None:
    """Download only if it is not already cached. Returns the snapshot path."""
    if is_downloaded(model_id):
        if handle is not None:
            info = get(model_id)
            handle.update(message=f"{info.label if info else model_id} already downloaded")
        return None
    return download_model(model_id, handle)

# style_transfer.py — apply the look of one picture to another
#
# color_transfer.py already moves COLOUR from a source to a target. This moves
# the rest of what people mean by "style": grain, brushwork, the character of
# the detail. Three methods, because they trade the same thing off differently.
#
#   texture     Non-generative. Keeps the target's pixels at low frequency and
#               borrows the source's detail at high frequency. Nothing is
#               invented, the composition is exactly preserved, and it runs on
#               a CPU in well under a second. The weakest effect of the three.
#
#   img2img     Diffusion, at low strength, guided by words CLIP finds for the
#               source. Preserves the target's structure but regenerates its
#               surface, so faces and text degrade. Needs sd-turbo, which
#               Inject Poetry already downloads.
#
#   ip_adapter  Diffusion, conditioned on the source IMAGE rather than on words
#               about it. The strongest stylisation and the only one that
#               transfers things language cannot name. Needs an extra adapter
#               checkpoint, and is the route that ControlNet later plugs into
#               for depth- and mask-aware transfer.
#
# All three take the same (source, target) pair the moodboard already calls
# [R] and [T], and all three return a PIL image at the target's own size.

from __future__ import annotations

import os

import numpy as np

try:
    import cv2
except ImportError:                                   # pragma: no cover
    cv2 = None

from PIL import Image

try:
    from . import gpu as _gpu
except ImportError:                                   # pragma: no cover
    import gpu as _gpu

# sd-turbo, the model Inject Poetry already fetches. Reusing it means the
# img2img route costs no extra download.
SD_MODEL = "stabilityai/sd-turbo"

# IP-Adapter for SD 1.5. sd-turbo is an SD-2.1 architecture, so the adapter has
# to match whichever base is used; see ip_adapter_transfer for the pairing.
IP_ADAPTER_REPO = "h94/IP-Adapter"
IP_BASE_MODEL = "runwayml/stable-diffusion-v1-5"

METHODS = ("texture", "img2img", "ip_adapter")


def _load_rgb(path_or_image) -> Image.Image:
    if isinstance(path_or_image, Image.Image):
        return path_or_image.convert("RGB")
    return Image.open(path_or_image).convert("RGB")


# --------------------------------------------------------------------------------------
# 1. Texture transfer — nothing invented
# --------------------------------------------------------------------------------------
def texture_transfer(source, target, strength: float = 0.6,
                     levels: int = 4, match_shape: bool = False) -> Image.Image:
    """
    Give the target the source's grain while keeping its own composition.

    Both pictures are split into a Laplacian pyramid: a small blurred residual
    holding structure and colour, and detail bands holding edges and texture at
    successively finer scales. The target keeps its own residual -- which is why
    the composition survives exactly -- and each of its detail bands is rescaled
    toward how much detail the SOURCE carries at that scale.

    Amplitude per scale, not the bands themselves. Blending the bands directly
    seems obvious and is wrong: Laplacian bands are spatially aligned, so mixing
    them paints the source's edges at the source's coordinates onto a different
    picture. Tried on two portraits, it ghosted one face over the other. Scaling
    instead means no source pixel is ever placed anywhere -- only a per-band,
    per-channel gain crosses over -- so ghosting is impossible by construction,
    and what transfers is how coarse or fine, how soft or crisp the source is.

    `match_shape` additionally matches each band's histogram, which carries the
    difference between grain that is even and grain that is spiky. It costs a
    full sort per band and is off by default because the effect is small beside
    the amplitude one.

    strength 0 returns the target untouched.
    """
    if cv2 is None:
        raise ImportError("OpenCV is required for texture transfer")

    strength = float(np.clip(strength, 0.0, 1.0))
    tgt = _load_rgb(target)
    src = _load_rgb(source).resize(tgt.size, Image.LANCZOS)
    if strength == 0.0:
        return tgt

    t = np.asarray(tgt, dtype=np.float32)
    s_img = np.asarray(src, dtype=np.float32)

    def pyramid(img: np.ndarray, n: int):
        """Detail bands, finest first, plus the residual they sit on."""
        bands, cur = [], img
        for _ in range(n):
            if min(cur.shape[:2]) < 8:
                break
            down = cv2.pyrDown(cur)
            up = cv2.pyrUp(down, dstsize=(cur.shape[1], cur.shape[0]))
            bands.append(cur - up)
            cur = down
        return bands, cur

    t_bands, t_res = pyramid(t, levels)
    s_bands, _ = pyramid(s_img, levels)

    mixed = []
    for tb, sb in zip(t_bands, s_bands):
        # Per channel, because the three carry different amounts of grain and a
        # single gain would tint the result.
        t_sd = tb.std(axis=(0, 1), keepdims=True)
        s_sd = sb.std(axis=(0, 1), keepdims=True)
        gain = s_sd / np.maximum(t_sd, 1e-5)
        # A source far coarser than the target would otherwise blow the band
        # out; 4x is already a dramatic change.
        gain = np.clip(gain, 0.1, 4.0)
        restyled = tb * gain

        if match_shape:
            for c in range(tb.shape[2]):
                restyled[:, :, c] = _match_distribution(restyled[:, :, c],
                                                        sb[:, :, c])

        mixed.append(tb * (1.0 - strength) + restyled * strength)

    # Rebuild on the TARGET's residual: its structure and colour, restyled detail.
    out = t_res
    for band in reversed(mixed):
        out = cv2.pyrUp(out, dstsize=(band.shape[1], band.shape[0])) + band

    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def _match_distribution(band: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """
    Give `band` the value distribution of `ref` while keeping its own ordering.

    Every pixel keeps its rank among its neighbours -- an edge stays where it
    was, and the strongest edges stay strongest -- but the values those ranks
    map to come from the reference. Optional, and costly: a full sort of every
    pixel in the band.
    """
    flat = band.ravel()
    order = np.argsort(flat, kind="stable")
    ranks = np.empty(flat.size, dtype=np.int64)
    ranks[order] = np.arange(flat.size)

    ref_sorted = np.sort(ref.ravel())
    if ref_sorted.size != flat.size:
        ref_sorted = np.quantile(ref_sorted, np.linspace(0.0, 1.0, flat.size))
    return ref_sorted[ranks].reshape(band.shape)


# --------------------------------------------------------------------------------------
# 2. Words for a picture, so img2img has something to aim at
# --------------------------------------------------------------------------------------
def describe(source, model_id: str | None = None, top_k: int = 6) -> str:
    """
    A short phrase describing the source, from Arcana's own label vocabulary.

    img2img is guided by text, so transferring a LOOK with it needs words for
    that look. Rather than ask the user to write them, score the source against
    the label list the app already embeds for naming clusters, and take the
    nearest few. It is a coarse description -- that is the honest limit of the
    img2img route, and the reason ip_adapter exists beside it.
    """
    try:
        from . import db as _db
        from . import models as _models
    except ImportError:                               # pragma: no cover
        import db as _db
        import models as _models

    model_id = model_id or _models.default_for("image").id

    src = _db.DEFAULT_LABELS.get("image")
    if not src or not os.path.exists(src):
        return ""
    texts, cache_base = _db._read_label_list(src)
    if not texts:
        return ""
    texts, mat = _db._encode_label_matrix(texts, "image", cache_base,
                                          model_id=model_id)
    if mat is None or not len(texts):
        return ""

    # img2vec_clip takes BGR, the way OpenCV hands images around this codebase.
    rgb = np.asarray(_load_rgb(source), dtype=np.uint8)
    bgr = rgb[:, :, ::-1].copy()
    vec = np.asarray(_db.img2vec_clip(bgr, model_id=model_id),
                     dtype=np.float32).ravel()
    vec /= max(float(np.linalg.norm(vec)), 1e-8)

    # The label matrix is already L2-normalised, so this is a cosine.
    scores = np.asarray(mat, dtype=np.float32) @ vec
    order = np.argsort(-scores)[:top_k]
    return ", ".join(str(texts[i]).lower() for i in order)


# --------------------------------------------------------------------------------------
# 3 & 4. The diffusion routes
# --------------------------------------------------------------------------------------
_PIPE_CACHE: dict = {}


def _sd_pipe(kind: str):
    """
    A diffusion pipeline, built once per process.

    Loading one costs seconds and several gigabytes, and the moodboard is a
    place people iterate, so the second transfer should not pay for it again.
    """
    hit = _PIPE_CACHE.get(kind)
    if hit is not None:
        return hit

    import torch
    from diffusers import StableDiffusionImg2ImgPipeline

    device = _gpu.device()
    # fp16 is a CUDA thing: most fp16 kernels are unimplemented on CPU, so
    # asking for the half variant there raises rather than merely being slow.
    dtype = torch.float16 if device == "cuda" else torch.float32
    repo = SD_MODEL if kind == "img2img" else IP_BASE_MODEL

    kw = {"torch_dtype": dtype}
    if device == "cuda" and kind == "img2img":
        kw["variant"] = "fp16"
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(repo, **kw).to(device)
    pipe.safety_checker = None
    pipe.watermark = None
    try:
        pipe.enable_vae_slicing()
    except Exception:
        pass

    if kind == "ip_adapter":
        pipe.load_ip_adapter(IP_ADAPTER_REPO, subfolder="models",
                             weight_name="ip-adapter_sd15.bin")
        # The adapter and its image encoder arrive AFTER the pipeline was moved
        # and cast, and they do not inherit either. Left alone that gives an
        # fp16 UNet talking to fp32 adapter weights, which fails at the first
        # cross-attention layer with "Input type (struct c10::Half) and bias
        # type (float) should be the same". Re-unify everything here rather
        # than trusting each component to have been built the same way.
        pipe = pipe.to(device, dtype)
        enc = getattr(pipe, "image_encoder", None)
        if enc is not None:
            pipe.image_encoder = enc.to(device=device, dtype=dtype)

    _PIPE_CACHE[kind] = pipe
    return pipe


def _fit_for_diffusion(img: Image.Image, max_side: int = 768) -> Image.Image:
    """Diffusion needs both sides divisible by 8, and hates very large inputs."""
    w, h = img.size
    scale = min(1.0, max_side / float(max(w, h)))
    nw = max(8, int(w * scale) // 8 * 8)
    nh = max(8, int(h * scale) // 8 * 8)
    return img.resize((nw, nh), Image.LANCZOS)


def img2img_transfer(source, target, strength: float = 0.45, steps: int = 4,
                     prompt: str | None = None, guidance: float = 1.0,
                     seed: int = 2222, progress=None) -> Image.Image:
    """
    Restyle the target with diffusion, aimed by words describing the source.

    Low strength keeps the target's composition; high strength drifts away from
    it. This route cannot see the source image -- only a description of it --
    so it transfers mood rather than a specific look. Use ip_adapter when the
    source's actual appearance is the point.
    """
    import torch

    tgt = _load_rgb(target)
    init = _fit_for_diffusion(tgt)
    text = prompt if prompt is not None else describe(source)

    if progress:
        progress(0.1, f"Describing the source: {text or '(no vocabulary)'}")
    pipe = _sd_pipe("img2img")
    if progress:
        progress(0.4, "Restyling")

    out = pipe(
        prompt=text or "a photograph",
        negative_prompt="text, letters, watermark, logo, lowres",
        image=init,
        strength=float(np.clip(strength, 0.05, 0.95)),
        num_inference_steps=int(max(1, steps)),
        guidance_scale=guidance,
        generator=torch.manual_seed(seed),
    ).images[0]
    return out.resize(tgt.size, Image.LANCZOS)


def ip_adapter_transfer(source, target, scale: float = 0.7, strength: float = 0.45,
                        steps: int = 12, guidance: float = 5.0, seed: int = 2222,
                        prompt: str = "", progress=None) -> Image.Image:
    """
    Restyle the target conditioned on the source IMAGE.

    IP-Adapter feeds the source through CLIP's image tower and injects it into
    the same cross-attention layers a prompt would use, so the model is steered
    by what the picture looks like rather than by words about it. `scale` is how
    loudly the source speaks: near 0 it is ignored, near 1 it overwhelms the
    target.

    This is the route ControlNet joins later: a depth map or segmentation mask
    becomes a second condition, which is what makes "apply this texture
    according to my target's depth" expressible.
    """
    import torch

    tgt = _load_rgb(target)
    init = _fit_for_diffusion(tgt)
    src = _load_rgb(source)

    if progress:
        progress(0.15, "Loading the adapter")
    pipe = _sd_pipe("ip_adapter")
    pipe.set_ip_adapter_scale(float(np.clip(scale, 0.0, 1.0)))
    if progress:
        progress(0.45, "Transferring style")

    out = pipe(
        prompt=prompt or "",
        negative_prompt="text, letters, watermark, logo, lowres",
        image=init,
        ip_adapter_image=src,
        strength=float(np.clip(strength, 0.05, 0.95)),
        num_inference_steps=int(max(1, steps)),
        guidance_scale=guidance,
        generator=torch.manual_seed(seed),
    ).images[0]
    return out.resize(tgt.size, Image.LANCZOS)


def transfer(method: str, source, target, progress=None, **kw) -> Image.Image:
    """One entry point, so the UI does not branch on method itself."""
    if method == "texture":
        return texture_transfer(source, target,
                                strength=kw.get("strength", 0.6),
                                levels=int(kw.get("levels", 4)),
                                match_shape=kw.get("match_shape", False))
    if method == "img2img":
        return img2img_transfer(source, target, progress=progress,
                                strength=kw.get("strength", 0.45),
                                steps=int(kw.get("steps", 4)),
                                prompt=kw.get("prompt"))
    if method == "ip_adapter":
        return ip_adapter_transfer(source, target, progress=progress,
                                   scale=kw.get("scale", 0.7),
                                   strength=kw.get("strength", 0.45),
                                   steps=int(kw.get("steps", 12)),
                                   prompt=kw.get("prompt", ""))
    raise ValueError(f"unknown style transfer method: {method!r}")


# --------------------------------------------------------------------------------------
# 5. ControlNet — style from one picture, structure from a map of another
# --------------------------------------------------------------------------------------
# IP-Adapter says WHAT it should look like. ControlNet says WHERE things go. Used
# together, the source's appearance is applied while the target's geometry is
# held in place -- which is what "apply this texture as a function of my
# target's depth" actually asks for.
#
# Each map answers a different question about the target:
#
#   depth      how far away each pixel is. Texture then follows the form
#              rather than being pasted flat across it.
#   canny      where the edges are. Holds outlines hard; the strictest.
#   luminance  how light each pixel is, used as a pseudo-depth. Needs no model
#              at all, and is a reasonable stand-in for a lit scene.

CONTROLNETS = {
    "depth": "lllyasviel/control_v11f1p_sd15_depth",
    "canny": "lllyasviel/control_v11p_sd15_canny",
    "luminance": "lllyasviel/control_v11f1p_sd15_depth",   # a depth-shaped map
}

DEPTH_MODEL = "Intel/dpt-hybrid-midas"

CONTROL_MAPS = ("depth", "canny", "luminance")

_DEPTH_PIPE = None


def _depth_map(img: Image.Image) -> Image.Image:
    """Estimate depth, as the 3-channel greyscale the depth ControlNet expects."""
    global _DEPTH_PIPE
    if _DEPTH_PIPE is None:
        from transformers import pipeline as _hf_pipeline
        _DEPTH_PIPE = _hf_pipeline("depth-estimation", model=DEPTH_MODEL,
                                   device=0 if _gpu.available() else -1)
    raw = _DEPTH_PIPE(img)["depth"]
    a = np.asarray(raw, dtype=np.float32)
    lo, hi = float(a.min()), float(a.max())
    a = (a - lo) / max(hi - lo, 1e-6)
    return Image.fromarray((np.stack([a] * 3, -1) * 255).astype(np.uint8))


def control_map(kind: str, image) -> Image.Image:
    """
    Build the map a ControlNet will read, from the TARGET.

    Returned as an ordinary image so the UI can show it: seeing the map is the
    difference between "the result looks wrong" and "the depth is wrong", and
    the second is something a user can act on.
    """
    img = _load_rgb(image)
    if kind == "depth":
        return _depth_map(img)
    if kind == "canny":
        if cv2 is None:
            raise ImportError("OpenCV is required for the canny map")
        g = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(g, 100, 200)
        return Image.fromarray(np.stack([edges] * 3, -1))
    if kind == "luminance":
        # Perceptual luminance, blurred so it reads as form rather than
        # texture, and inverted: nearer things in a lit scene are usually
        # brighter, and depth maps put nearer at the bright end too.
        a = np.asarray(img, dtype=np.float32)
        lum = 0.2126 * a[:, :, 0] + 0.7152 * a[:, :, 1] + 0.0722 * a[:, :, 2]
        if cv2 is not None:
            k = max(3, (min(img.size) // 64) | 1)
            lum = cv2.GaussianBlur(lum, (k, k), 0)
        lo, hi = float(lum.min()), float(lum.max())
        lum = (lum - lo) / max(hi - lo, 1e-6)
        return Image.fromarray((np.stack([lum] * 3, -1) * 255).astype(np.uint8))
    raise ValueError(f"unknown control map: {kind!r}")


_CN_CACHE: dict = {}


def _controlnet_pipe(kind: str):
    """
    A ControlNet img2img pipeline with IP-Adapter attached, built once.

    Same SD-1.5 base as ip_adapter_transfer, so the two share their download.
    """
    hit = _CN_CACHE.get(kind)
    if hit is not None:
        return hit

    import torch
    from diffusers import (ControlNetModel,
                           StableDiffusionControlNetImg2ImgPipeline)

    device = _gpu.device()
    dtype = torch.float16 if device == "cuda" else torch.float32

    net = ControlNetModel.from_pretrained(CONTROLNETS[kind], torch_dtype=dtype)
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        IP_BASE_MODEL, controlnet=net, torch_dtype=dtype)
    pipe.safety_checker = None
    pipe.watermark = None
    pipe.load_ip_adapter(IP_ADAPTER_REPO, subfolder="models",
                         weight_name="ip-adapter_sd15.bin")
    # Cast AFTER the adapter and its image encoder are attached: neither
    # inherits the pipeline's dtype, and an fp16 UNet talking to fp32 adapter
    # weights fails at the first cross-attention layer.
    pipe = pipe.to(device, dtype)
    enc = getattr(pipe, "image_encoder", None)
    if enc is not None:
        pipe.image_encoder = enc.to(device=device, dtype=dtype)
    try:
        pipe.enable_vae_slicing()
    except Exception:
        pass

    _CN_CACHE[kind] = pipe
    return pipe


def controlnet_transfer(source, target, control: str = "depth",
                        scale: float = 0.7, control_scale: float = 0.8,
                        strength: float = 0.7, steps: int = 16,
                        guidance: float = 5.0, seed: int = 2222,
                        prompt: str = "", progress=None):
    """
    Apply the look of the source while the target's geometry holds it in place.

    Two conditions at once. IP-Adapter carries the source's appearance;
    ControlNet carries a map of the target -- its depth, its edges, its
    luminance -- and holds the generation to it. That is the difference between
    a texture pasted flat over a picture and one that follows its form.

    `scale` is how loudly the source speaks, `control_scale` how strictly the
    map is obeyed. They pull against each other: a high control_scale with a
    high strength keeps the composition while still restyling it, which is
    usually what is wanted. Turning control_scale down approaches plain
    ip_adapter_transfer.

    Returns (image, map) so the caller can show what the geometry actually was.
    """
    import torch

    if control not in CONTROLNETS:
        raise ValueError(f"unknown control map: {control!r}")

    tgt = _load_rgb(target)
    init = _fit_for_diffusion(tgt)
    src = _load_rgb(source)

    if progress:
        progress(0.1, f"Building the {control} map")
    cmap = control_map(control, init).resize(init.size, Image.LANCZOS)

    if progress:
        progress(0.3, f"Loading ControlNet ({control})")
    pipe = _controlnet_pipe(control)
    pipe.set_ip_adapter_scale(float(np.clip(scale, 0.0, 1.0)))

    if progress:
        progress(0.55, "Transferring")
    out = pipe(
        prompt=prompt or "",
        negative_prompt="text, letters, watermark, logo, lowres",
        image=init,
        control_image=cmap,
        ip_adapter_image=src,
        strength=float(np.clip(strength, 0.05, 0.95)),
        num_inference_steps=int(max(1, steps)),
        guidance_scale=guidance,
        controlnet_conditioning_scale=float(np.clip(control_scale, 0.0, 2.0)),
        generator=torch.manual_seed(seed),
    ).images[0]
    return out.resize(tgt.size, Image.LANCZOS), cmap

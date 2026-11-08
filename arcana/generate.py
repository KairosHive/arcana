# generate.py -- pictures made by sd-turbo, outside any UI
#
# Inject Poetry used to build its diffusion pipeline inside a Dash job. It is
# here now so the `generate` op, the app and an agent share one pipeline per
# process, loaded once. sd-turbo is a distilled model: four steps and no
# classifier-free guidance give a usable picture, which is what makes a CPU
# run tolerable (about half a minute a picture) and a GPU run near instant.
#
# Everything made here is marked `generated` in the ledger and on any board,
# so it can never pass for a photograph from the archive.

from __future__ import annotations

import threading

from PIL import Image

SD_MODEL = "stabilityai/sd-turbo"
NEGATIVE = "text, letters, watermark, logo, blurry, low quality"

_PIPES: dict = {}
_LOCK = threading.Lock()


def _gpu():
    try:
        from . import gpu as _g
    except ImportError:                               # pragma: no cover
        import gpu as _g
    return _g


def pipe(kind: str):
    """The txt2img or img2img pipeline, built once per process and sharing weights."""
    with _LOCK:
        hit = _PIPES.get(kind)
        if hit is not None:
            return hit
        import torch
        from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline
        device = _gpu().device()
        base = _PIPES.get("txt2img")
        if base is None:
            # fp16 is a CUDA thing: on the CPU most fp16 kernels are missing,
            # and asking for them raised on every machine without a card.
            kw = ({"torch_dtype": torch.float16, "variant": "fp16"} if device == "cuda"
                  else {"torch_dtype": torch.float32})
            base = StableDiffusionPipeline.from_pretrained(SD_MODEL, **kw).to(device)
            try:
                base.enable_xformers_memory_efficient_attention()
            except Exception:
                pass                                   # CUDA-only and optional
            base.vae.enable_slicing()
            base.safety_checker = None
            _PIPES["txt2img"] = base
        if kind == "img2img":
            _PIPES["img2img"] = StableDiffusionImg2ImgPipeline(**base.components)
        return _PIPES[kind]


def fit(img: Image.Image, max_side: int = 1024) -> Image.Image:
    """A copy no larger than max_side, with sides a multiple of 8 as the VAE needs."""
    w, h = img.size
    scale = min(1.0, max_side / float(max(w, h)))
    nw, nh = max(8, int(w * scale) // 8 * 8), max(8, int(h * scale) // 8 * 8)
    return img.convert("RGB").resize((nw, nh), Image.LANCZOS)


def _callback(progress, steps):
    if progress is None:
        return None

    def cb(pipe_ref, step, timestep, kwargs):
        progress((step + 1) / float(steps), f"step {step + 1} of {steps}")
        return kwargs
    return cb


def img2img(init: Image.Image, prompt: str, strength: float = 0.72, steps: int = 4,
            seed: int = 2222, negative: str = NEGATIVE, progress=None) -> Image.Image:
    """`init` redrawn toward `prompt`. strength 0 keeps it, 1 replaces it."""
    import torch
    p = pipe("img2img")
    return p(prompt=prompt, negative_prompt=negative, image=fit(init),
             strength=float(max(0.0, min(1.0, strength))), num_inference_steps=int(steps),
             guidance_scale=1.0, generator=torch.manual_seed(int(seed)),
             callback_on_step_end=_callback(progress, steps)).images[0]


def txt2img(prompt: str, steps: int = 4, seed: int = 2222, width: int = 512,
            height: int = 512, negative: str = NEGATIVE, progress=None) -> Image.Image:
    """A picture from words alone. sd-turbo was trained at 512 px."""
    import torch
    p = pipe("txt2img")
    return p(prompt=prompt, negative_prompt=negative, num_inference_steps=int(steps),
             guidance_scale=0.0, width=int(width) // 8 * 8, height=int(height) // 8 * 8,
             generator=torch.manual_seed(int(seed)),
             callback_on_step_end=_callback(progress, steps)).images[0]


def release() -> None:
    """Free the card: move the pipelines off it and drop them."""
    with _LOCK:
        for p in _PIPES.values():
            try:
                p.to("cpu")
            except Exception:
                pass
        _PIPES.clear()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

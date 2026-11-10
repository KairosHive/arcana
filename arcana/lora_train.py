# lora_train.py -- a texture LoRA trained on a set, for Alchemy's texture LoRA method
#
# Ported from latent-lab's lora.py, keeping what its runs taught:
#
#   * crops see a third of a picture's short side (`detail` 3) and are drawn at
#     512: squeezed whole into 512 pixels, a photograph's grain is gone before
#     the model sees it;
#   * every picture is captioned on its own (captions.py: material, surface,
#     colour and condition) with the trigger after: one caption for a whole set
#     drew "beaded confetti", captions per picture fixed it;
#   * checkpoints are numbered and kept, each with a sheet drawn at fixed seeds,
#     because the loss does not say when to stop: blur sets in near
#     steps x strength = 4800, and the checkpoint is chosen by eye;
#   * the adapters train in fp32 over a half-precision model, a step whose
#     gradients are not finite is dropped, and the VAE encodes in fp32.
#
# A run writes <data dir>/loras/<name>/step-NNNNNN/ (pytorch_lora_weights.safetensors,
# sheet.jpg, arcana-lora.json), which loras.py finds and the texture LoRA
# method lists as <name>/step-NNNNNN. The base is the SD 1.5 the ControlNet and
# IP-Adapter pipelines draw on, so the LoRA fits the method it is made for.

from __future__ import annotations

import json
import os
import queue
import random
import threading
import time

import numpy as np
from . import hub as _hub

BASE = "runwayml/stable-diffusion-v1-5"      # style_transfer.IP_BASE_MODEL
TARGETS = ["to_k", "to_q", "to_v", "to_out.0"]
SHEET_SEEDS = (11, 22, 33, 44)
META = "arcana-lora.json"


def loras_root() -> str:
    from . import paths as _paths

    return _paths.ensure_dir(os.path.join(_paths.data_dir(), "loras"))


def trigger_for(name: str) -> str:
    return f"in {name} style"


def _square(path: str, size: int, detail: float, rng: random.Random):
    """One training crop: square, 1/detail of the short side, flipped at random, in [-1, 1]."""
    from PIL import Image, ImageOps

    side = max(size, round(size * detail))
    with Image.open(path) as im:
        shrink = min(im.size) / side
        if shrink >= 2:
            im.draft("RGB", (round(im.width / shrink), round(im.height / shrink)))
        im = ImageOps.exif_transpose(im).convert("RGB")
    scale = side / min(im.size)
    im = im.resize((max(side, round(im.width * scale)), max(side, round(im.height * scale))), Image.LANCZOS)
    x, y = rng.randint(0, im.width - size), rng.randint(0, im.height - size)
    im = im.crop((x, y, x + size, y + size))
    if rng.random() < 0.5:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    if rng.random() < 0.5:
        im = im.transpose(Image.FLIP_TOP_BOTTOM)   # a texture has no up
    return np.asarray(im, np.float32) / 127.5 - 1.0


def captions(paths: list[str], trigger: str, progress=None) -> dict[str, str]:
    """A caption per picture (captions.py): material, surfaces, colours, condition, then the trigger."""
    from . import captions as _captions

    return _captions.caption_all(paths, trigger, progress)


def _free_card():
    """Let Alchemy's models go: training wants most of the card."""
    from . import style_transfer as st
    from . import ip_labels

    st._PIPE_CACHE.clear()
    st._CN_CACHE.clear()
    ip_labels.release_encoder()
    try:
        from .alchemy.memory import MODELS, GB
        MODELS.room(gpu=12 * GB, protect=0)
    except Exception:
        pass
    import gc
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


def train(name: str, paths: list[str], steps: int = 2000, rank: int = 16, lr: float = 1e-4, batch: int = 2,
          detail: float = 3.0, size: int = 512, every: int = 500, origin: dict | None = None,
          progress=None, cancelled=None) -> dict:
    """Train a LoRA on `paths`; returns {"name", "folder", "checkpoints", "trigger", ...}."""
    import torch
    import torch.nn.functional as F

    from . import gpu as _gpu

    if _gpu.device() != "cuda":
        raise RuntimeError("Training a LoRA needs an NVIDIA GPU with CUDA. This Arcana runs on the CPU.")
    paths = [p for p in paths if os.path.isfile(p)]
    if len(paths) < 8:
        raise ValueError(f"A LoRA needs at least 8 pictures; this set has {len(paths)} on disk.")
    slug = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in name.strip().lower()).strip("-") or "lora"
    root = os.path.join(loras_root(), slug)
    os.makedirs(root, exist_ok=True)
    trigger = trigger_for(slug)
    t0 = time.time()

    def say(f, msg):
        if progress:
            progress(f, msg)

    say(0.01, "captioning the pictures")
    caps = captions(paths, trigger, progress)
    say(0.08, "loading Stable Diffusion 1.5")
    _free_card()

    from diffusers import DDPMScheduler, StableDiffusionPipeline
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict

    device = torch.device("cuda")
    half = torch.float16
    pipe = _hub.pretrained(StableDiffusionPipeline.from_pretrained, BASE, torch_dtype=half, safety_checker=None)
    pipe.to(device)
    unet, vae = pipe.unet, pipe.vae
    vae.to(torch.float32)
    for part in (unet, vae, pipe.text_encoder):
        part.requires_grad_(False)
    unet.add_adapter(LoraConfig(r=rank, lora_alpha=rank, init_lora_weights="gaussian", target_modules=TARGETS))
    trained = [q for q in unet.parameters() if q.requires_grad]
    for q in trained:
        q.data = q.data.float()
    noise_scheduler = _hub.pretrained(DDPMScheduler.from_pretrained, BASE, subfolder="scheduler")
    optimiser = torch.optim.AdamW(trained, lr=lr, weight_decay=1e-2)
    said: dict = {}

    def embedding(text):
        if text not in said:
            with torch.no_grad():
                said[text] = pipe.encode_prompt(text, device, 1, False)[0]
        return said[text]

    # Crops are cut on a thread while the card trains: a 24-megapixel JPEG costs about as long to open
    # as a training step, and DataLoader workers are processes that a server cannot spawn safely.
    batches: queue.Queue = queue.Queue(maxsize=6)
    stop = threading.Event()

    def feed():
        rng = random.Random(0)
        while not stop.is_set():
            pix, texts = [], []
            while len(pix) < batch:
                p = rng.choice(paths)
                try:
                    pix.append(_square(p, size, detail, rng))
                    texts.append(caps[p])
                except OSError:
                    continue
            item = (np.stack(pix), texts)
            while not stop.is_set():
                try:
                    batches.put(item, timeout=1)
                    break
                except queue.Full:
                    continue

    feeder = threading.Thread(target=feed, daemon=True)
    feeder.start()

    checkpoints = []
    from . import captions as _captions

    sample_prompt = _captions.typical(caps, trigger)   # the sheets and the method prompt with this

    def keep(step):
        out = os.path.join(root, f"step-{step:06d}")
        os.makedirs(out, exist_ok=True)
        pipe.__class__.save_lora_weights(
            save_directory=out,
            unet_lora_layers=convert_state_dict_to_diffusers(get_peft_model_state_dict(unet)))
        sheet(pipe, vae, sample_prompt, device).save(os.path.join(out, "sheet.jpg"), quality=90)
        meta = {"base": BASE, "trigger": trigger, "step": step, "steps": steps, "rank": rank, "lr": lr,
                "batch": batch, "detail": detail, "size": size, "pictures": len(paths), "origin": origin or {},
                "prompt": sample_prompt, "seconds": round(time.time() - t0)}
        with open(os.path.join(out, META), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=1)
        checkpoints.append(f"{slug}/step-{step:06d}")

    losses = []
    # The app switches gradients off for the whole process at import (no search builds a graph);
    # training is the one place that needs them, and only on this thread.
    grad = torch.enable_grad()
    grad.__enter__()
    try:
        for step in range(1, steps + 1):
            if cancelled and cancelled():
                break
            pixels, texts = batches.get()
            pixels = torch.from_numpy(pixels).permute(0, 3, 1, 2).to(device, torch.float32)
            with torch.no_grad():
                latents = vae.encode(pixels).latent_dist.sample() * vae.config.scaling_factor
            latents = latents.to(half)
            told = torch.cat([embedding(t) for t in texts])
            noise = torch.randn_like(latents)
            when = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
            noisy = noise_scheduler.add_noise(latents, noise, when)
            got = unet(noisy, when, encoder_hidden_states=told).sample
            loss = F.mse_loss(got.float(), noise.float())
            optimiser.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            if not torch.isfinite(torch.nn.utils.clip_grad_norm_(trained, 1.0)):
                continue
            optimiser.step()
            losses.append(float(loss.item()))
            if step % 10 == 0:
                rate = step / max(1e-6, time.time() - t0)
                left = (steps - step) / max(rate, 1e-6)
                say(0.1 + 0.88 * step / steps,
                    f"step {step} of {steps}, loss {np.mean(losses[-50:]):.4f}, about {int(left // 60)} min left")
            if step % every == 0 or step == steps:
                say(0.1 + 0.88 * step / steps, f"step {step}: keeping a checkpoint and drawing its sheet")
                keep(step)
        if not checkpoints and losses:
            keep(len(losses))
    finally:
        grad.__exit__(None, None, None)
        stop.set()
        del pipe, unet, vae, optimiser
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    return {"name": slug, "folder": root, "checkpoints": checkpoints, "trigger": trigger,
            "pictures": len(paths), "seconds": round(time.time() - t0),
            "stopped": bool(cancelled and cancelled())}


def sheet(pipe, vae, prompt: str, device, size: int = 384):
    """Four draws at fixed seeds: the same seeds at every checkpoint, so checkpoints compare by eye."""
    import torch
    from PIL import Image

    pipe.set_progress_bar_config(disable=True)
    tiles = []
    for seed in SHEET_SEEDS:
        with torch.no_grad():
            lat = pipe(prompt=prompt, negative_prompt="text, letters, watermark, lowres", num_inference_steps=20,
                       guidance_scale=7.0, height=size, width=size,
                       generator=torch.Generator(device).manual_seed(seed), output_type="latent").images
            rgb = vae.decode(lat.float() / vae.config.scaling_factor).sample
        tiles.append(((rgb[0].clamp(-1, 1) + 1) * 127.5).permute(1, 2, 0).round().byte().cpu().numpy())
    return Image.fromarray(np.concatenate(tiles, 1))


def runs() -> list[dict]:
    """Every LoRA run under <data dir>/loras with its checkpoints and sheets, newest first."""
    root = loras_root()
    out = []
    for name in os.listdir(root):
        folder = os.path.join(root, name)
        if not os.path.isdir(folder):
            continue
        cps = []
        for step in sorted(os.listdir(folder)):
            d = os.path.join(folder, step)
            if not os.path.isfile(os.path.join(d, "pytorch_lora_weights.safetensors")):
                continue
            meta = {}
            try:
                with open(os.path.join(d, META), encoding="utf-8") as f:
                    meta = json.load(f)
            except (OSError, ValueError):
                pass
            sheet_path = os.path.join(d, "sheet.jpg")
            cps.append({"name": f"{name}/{step}", "step": meta.get("step"), "sheet": sheet_path if os.path.isfile(sheet_path) else None,
                        "trigger": meta.get("trigger"), "when": os.path.getmtime(d)})
        if cps:
            out.append({"name": name, "checkpoints": cps, "when": max(c["when"] for c in cps)})
    out.sort(key=lambda r: r["when"], reverse=True)
    return out

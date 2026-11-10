"""
Does a learned look (LoRA) beat an IP label measured on the same set? The three tests of 2026-11-09.

    uv run python evals/lora_vs_label.py --set water-reflections --lora water-reflections/step-001000 \\
        --picture <a photograph> --face <a portrait> --out output/learn/lora-eval

1. Drawing from nothing. Eight draws per row at 512, no source picture: the base model prompted with the
   set's typical caption without the trigger, the label alone, the LoRA alone, both. Beside eight crops of
   the set at the scale the LoRA trained on.
2. Whose look. Each row's draws scored by mean cosine to the set's centre (one crop of each picture) in
   CLIP ViT-H (the label's own space) and in DINOv2 (a space the label never uses), and by fine detail
   (mean absolute Laplacian, 0-255) against the set's own.
3. Grain at 100%. A picture and a face redrawn by Alchemy's learned-look method with the label, the
   LoRA, and both, cropped at one pixel per pixel.

What the first run found (water reflections, 120 pictures, SD 1.5, 1000 steps at 1e-4): the label alone
sat at 0.78 in DINOv2 against 0.81 for real members; the LoRA at 0.40, below the base model's stock water
(0.51). The LoRA carried the set's softness and palette and lost its ripple structure. Writes the draws
and tests.json to --out.
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
from PIL import Image, ImageOps

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arcana import boards, ip_labels, loras  # noqa: E402


def crop_third(path, size=512, seed=0):
    rng = random.Random(seed)
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
    side = int(min(im.size) / 3)
    x, y = rng.randint(0, im.width - side), rng.randint(0, im.height - side)
    return im.crop((x, y, x + side, y + side)).resize((size, size), Image.LANCZOS)


def detail(im):
    import cv2
    g = np.asarray(im.convert("L"), np.float32)
    return float(np.abs(cv2.Laplacian(g, cv2.CV_32F)).mean())


def unit(m):
    return m / np.linalg.norm(m, axis=-1, keepdims=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--set", required=True, help="The set the label and the LoRA were made from.")
    ap.add_argument("--lora", required=True, help="The LoRA checkpoint, as Alchemy lists it (name/step-NNNNNN).")
    ap.add_argument("--label", default=None, help="The IP label; defaults to the set's own.")
    ap.add_argument("--picture", default=None, help="A photograph for the grain test.")
    ap.add_argument("--face", default=None, help="A portrait for the grain test on a face.")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--out", default=os.path.join("output", "learn", "lora-eval"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    label = a.label or ip_labels.slugify(a.set)
    lab = ip_labels.load(label)
    paths, _ = boards.present(boards.load(a.set))
    cap = loras.prompt(a.lora) or loras.trigger(a.lora)
    trigger = loras.trigger(a.lora)
    plain = cap.replace(", " + trigger, "").replace(trigger, "").strip(", ") or "a close-up photograph"

    from diffusers import StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16,
                                                   safety_checker=None).to("cuda")
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
    pipe = pipe.to("cuda", torch.float16)
    pipe.image_encoder = pipe.image_encoder.to("cuda", torch.float16)
    pipe.load_lora_weights(str(loras.locate(a.lora)), weight_name=loras.WEIGHTS, adapter_name="look")
    pipe.set_progress_bar_config(disable=True)

    def draw(prompt, use_label, use_lora, seed):
        (pipe.enable_lora(), pipe.set_adapters(["look"], [1.0])) if use_lora else pipe.disable_lora()
        v = ip_labels.point(lab, push=2.0, spread=0.25, seed=seed) if use_label else np.zeros(ip_labels.DIM, np.float32)
        pipe.set_ip_adapter_scale(0.8 if use_label else 0.0)
        emb = torch.tensor(v, dtype=torch.float16, device="cuda").view(1, 1, -1)
        emb = torch.cat([torch.zeros_like(emb), emb])
        return pipe(prompt=prompt, negative_prompt="text, letters, watermark, lowres", ip_adapter_image_embeds=[emb],
                    num_inference_steps=25, guidance_scale=7.0, height=512, width=512,
                    generator=torch.Generator("cuda").manual_seed(seed)).images[0]

    rows = {"base": (plain, False, False), "label": ("a close-up photograph", True, False),
            "lora": (cap, False, True), "both": (cap, True, True)}
    draws = {"photos": [crop_third(p, seed=i) for i, p in enumerate(random.Random(7).sample(paths, min(a.seeds, len(paths))))]}
    for key, (prompt, use_label, use_lora) in rows.items():
        draws[key] = [draw(prompt, use_label, use_lora, 101 + i) for i in range(a.seeds)]
    for key, ims in draws.items():
        for i, im in enumerate(ims):
            im.save(os.path.join(a.out, f"N-{key}-{i}.jpg"), quality=90)
    del pipe
    torch.cuda.empty_cache()

    held = [crop_third(p, seed=1000 + i) for i, p in enumerate(paths)]
    from transformers import AutoImageProcessor, Dinov2WithRegistersModel
    dproc = AutoImageProcessor.from_pretrained("facebook/dinov2-with-registers-base")
    dino = Dinov2WithRegistersModel.from_pretrained("facebook/dinov2-with-registers-base").to("cuda").eval()

    def dino_vecs(ims):
        out = []
        for i in range(0, len(ims), 16):
            px = dproc(images=ims[i:i + 16], return_tensors="pt").pixel_values.to("cuda")
            with torch.no_grad():
                h = dino(pixel_values=px).last_hidden_state
            out.append(torch.cat([h[:, 0], h[:, 5:].mean(1)], 1).float().cpu().numpy())
        return np.concatenate(out)

    scores = {}
    for space, enc in (("clip", ip_labels.encode), ("dino", dino_vecs)):
        centre = unit(unit(enc(held)).mean(0))
        scores[space] = {k: round(float((unit(enc(v)) @ centre).mean()), 4) for k, v in draws.items()}
    results = {"set": a.set, "lora": a.lora, "label": label, "caption": cap, "scores": scores,
               "detail": {k: round(float(np.mean([detail(im) for im in v])), 2) for k, v in draws.items()},
               "set_detail": round(float(np.mean([detail(im) for im in held[:40]])), 2), "when": time.time()}
    del dino
    torch.cuda.empty_cache()

    from arcana.alchemy import pipeline as pl
    grain = {}
    for name, pic, mask in (("picture", a.picture, None), ("face", a.face, {"layers": ["faces"], "feather": 4})):
        if not pic:
            continue
        for key, k in (("label", {"lora": "none", "label": label, "material": 0.8, "prompt": "a close-up photograph"}),
                       ("lora", {"lora": a.lora, "material": 0.0}),
                       ("both", {"lora": a.lora, "label": label, "material": 0.8})):
            spec = {"picture": pic, "method": "generative.lora", "knobs": {"strength": 0.6, "colour": 0.8, "lora_w": 1.0,
                                                                            "seed": 1, **k}}
            if mask:
                spec["mask"] = mask
            im, _ = pl.run(spec, None if mask else 768)
            w, h = im.size
            c = im.crop((w // 2 - 200, h // 2 - 200, w // 2 + 200, h // 2 + 200))
            c.save(os.path.join(a.out, f"G-{name}-{key}-100.png"))
            grain[f"{name}-{key}"] = round(detail(c), 2)
    results["grain"] = grain
    with open(os.path.join(a.out, "tests.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()

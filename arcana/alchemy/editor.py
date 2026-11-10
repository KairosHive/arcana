"""The material editor: a masked part of a picture redrawn by an instruction-following editor model.

Ported from latent-lab's editor.py, the method behind its atlas (34 cards: a face turned to cracked
paint, a headscarf to peeling paint, a cliff to pine bark, clothing to rippled reflection).

FLUX.2 [klein] 4B (Black Forest Labs, Apache 2.0) edits by looking: given the picture, a mask and a
REFERENCE image, its inpainting pipeline redraws the masked part with the reference in view and an
instruction in words. It understands "replace the fabric of the dress with this" the way an editor
model does, the reference's material put onto the thing's own folds, seams and light, where an
IP-Adapter only hears an embedding.

The reference is a SWATCH: a texture photograph (the material [1]), or a place in IP space drawn
first by Stable Diffusion 1.5 from an IP label (the lab's "space" swatch).

What the lab measured and this keeps:
  * the image model (8 GB in bf16) and VAE are read from disk straight onto the card, and the 8 GB
    text encoder only for an instruction not read yet, with the image model out of the way, then let
    go; what it makes of an instruction is kept. The lab kept both in RAM; here RAM and page file are
    mostly committed by other programs, and staging 8 GB through them is refused;
  * the swatch is read at 512 px, the words are padded to 128 tokens, full quality is 6 steps (8
    looked the same and cost 22% more), 4 steps for a draft, no guidance (the model is distilled);
  * when the mask holds only part of the frame the editor works AROUND it, the mask's extent with a
    margin of context, drawn at no less than 640 px and laid back, which is faster and gives a small
    face more pixels than it had;
  * "make the masked surfaces out of ..." left a dress as it was; naming the thing ("replace the
    surface of the dress with ...") changes it, so the instruction names what the mask holds.
"""

from __future__ import annotations

import collections
import math
import threading

import numpy as np
from PIL import Image
from .. import hub as _hub

KLEIN = "black-forest-labs/FLUX.2-klein-4B"
KEY = "editor: FLUX.2 klein"
INSTRUCTION = ("Replace the surface of {what} with the material from the reference image: its texture, grain "
               "and colour, following the shape, folds and lighting of {what}.")
SWATCH_SIDE = 512


def instruction(what: str | None) -> str:
    return INSTRUCTION.format(what=(what or "").strip() or "the masked area")


# What each mask preset holds, in words the editor can act on.
PRESET_WORDS = {"faces": "the face", "people": "the person's clothes", "clothes": "the clothes",
                "subject": "the main subject", "sky": "the sky", "water": "the water", "foliage": "the foliage",
                "ground": "the ground", "rock and stone": "the rock", "buildings": "the building",
                "background": "the background"}


def mask_words(mask) -> str | None:
    """What a mask spec's first layer selects, as words: "thing:the coat" gives "the coat", the
    faces preset "the face". None for a mask that names nothing (light, depth, a gradient)."""
    if not mask:
        return None
    if isinstance(mask, str):
        layers = [mask]
    elif isinstance(mask, (list, tuple)):
        layers = list(mask)
    else:
        layers = list(mask.get("layers") or [])
    if not layers:
        return None
    first = str(layers[0]).strip()
    for head in ("thing:", "prompt:", "stuff:"):
        if first.startswith(head):
            words = first[len(head):].strip()
            if not words:
                return None
            return words if words.lower().startswith(("the ", "a ", "an ")) else "the " + words
    return PRESET_WORDS.get(first)


HELD = ["latents", "image_latents", "noise", "timesteps"]


def around(mask, margin=0.3, most=0.6):
    """Where to work when the mask holds only part of the picture: its extent with a margin of
    context on every side, or None when that would be most of the frame anyway."""
    h, w = mask.shape
    ys, xs = np.nonzero(mask > 0.02)
    if not len(xs):
        return None
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    mx, my = max(48.0, (x1 - x0) * margin), max(48.0, (y1 - y0) * margin)
    x0, x1 = max(0, int(x0 - mx)), min(w, int(x1 + mx))
    y0, y1 = max(0, int(y0 - my)), min(h, int(y1 + my))
    return None if (x1 - x0) * (y1 - y0) > most * w * h else (x0, y0, x1, y1)


def worksize(box, frame, least=640):
    """The size the surroundings are drawn at: the frame's own density, raised so the long side is at
    least `least` (a small face gets more pixels than it had), never past the frame's long side."""
    cw, ch = box[2] - box[0], box[3] - box[1]
    k = min(max(1.0, least / max(cw, ch)), max(frame) / max(cw, ch))
    return max(16, round(cw * k / 16) * 16), max(16, round(ch * k / 16) * 16)


def resized(a, w, h):
    return np.asarray(Image.fromarray(np.ascontiguousarray(a, np.float32), "F").resize((w, h), Image.BILINEAR))


class Editor:
    def __init__(self):
        self.pipe = None
        self.lock = threading.Lock()
        self.read: collections.OrderedDict = collections.OrderedDict()   # instruction -> embeddings
        self.parked = True

    @property
    def device(self):
        import torch
        return torch.device("cuda")

    def _transformer(self):
        """The image model, read onto the card a tensor at a time (to_card)."""
        from diffusers import Flux2Transformer2DModel

        return to_card(Flux2Transformer2DModel, "transformer")

    def load(self):
        if self.pipe is None:
            import torch
            from diffusers import AutoencoderKLFlux2, Flux2KleinInpaintPipeline, FlowMatchEulerDiscreteScheduler
            from transformers import AutoTokenizer

            from .memory import GB, LOAD, MODELS

            device = self.device

            class Pipe(Flux2KleinInpaintPipeline):
                # diffusers takes the device of the first component by name, the text encoder, which is
                # absent here; the pictures must go where the VAE and the image model run.
                @property
                def _execution_device(self):
                    return device

            _free_diffusion()
            with LOAD:
                MODELS.room(gpu=9 * GB, protect=0)
                pipe = Pipe(scheduler=_hub.pretrained(FlowMatchEulerDiscreteScheduler.from_pretrained, KLEIN, subfolder="scheduler"),
                            vae=_hub.pretrained(AutoencoderKLFlux2.from_pretrained, KLEIN, subfolder="vae", torch_dtype=torch.bfloat16,
                                                                   device_map="cuda"),
                            text_encoder=None, tokenizer=_hub.pretrained(AutoTokenizer.from_pretrained, KLEIN, subfolder="tokenizer"),
                            transformer=self._transformer(), is_distilled=True)
            pipe.set_progress_bar_config(disable=True)
            pipe._callback_tensor_inputs = HELD
            self.pipe, self.parked = pipe, False
            MODELS.add(KEY, self.unload, self.lock, ram=0, gpu=int(8.5 * GB))
        elif self.parked:
            self.pipe.transformer = self._transformer()
            self.parked = False
        from .memory import MODELS
        MODELS.used(KEY)
        return self.pipe

    def unload(self):
        import gc

        self.pipe = None
        self.read.clear()
        self.parked = True
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    def _drop_transformer(self):
        import gc

        import torch

        if self.pipe is not None and not self.parked:
            self.pipe.transformer = None
            self.parked = True
            gc.collect()
            torch.cuda.empty_cache()

    def park(self):
        """The image model off the card, for another pipeline's room; the next edit reads it back from disk."""
        if self.pipe is None or self.parked:
            return
        with self.lock:
            self._drop_transformer()

    def words(self, pipe, prompt, length=128):
        """The instruction's embeddings. A new instruction reads the text encoder (Qwen3 4B) onto the card with
        the image model out of the way, runs it, and lets it go; the embeddings are kept."""
        import gc

        import torch
        from transformers import Qwen3ForCausalLM

        key = (prompt, int(length))
        if key not in self.read:
            self._drop_transformer()
            encoder = to_card(Qwen3ForCausalLM, "text_encoder")
            try:
                pipe.text_encoder = encoder
                self.read[key] = pipe.encode_prompt(prompt=prompt, device=self.device,
                                                    max_sequence_length=int(length))[0].detach()
            finally:
                pipe.text_encoder = None
                del encoder
                gc.collect()
                torch.cuda.empty_cache()
            while len(self.read) > 32:
                self.read.popitem(last=False)
        self.read.move_to_end(key)
        return self.read[key]

    def generate(self, prompt, references=(), size=(768, 768), steps=6, seed=0):
        """A picture drawn from nothing by FLUX.2 klein, from words and, optionally, reference pictures it
        looks at (a set's own photographs). Shares every weight with the editor: nothing more is loaded."""
        import torch
        from diffusers import Flux2KleinPipeline

        w, h = (int(size[0]) // 16 * 16, int(size[1]) // 16 * 16)
        refs = []
        for r in references:
            r = r.convert("RGB")
            r.thumbnail((SWATCH_SIDE, SWATCH_SIDE), Image.LANCZOS)
            refs.append(r)
        with self.lock, torch.no_grad():
            pipe = self.load()
            embeds = self.words(pipe, prompt, 128 if len(prompt.split()) <= 80 else 512)
            pipe = self.load()
            drawer = Flux2KleinPipeline(scheduler=pipe.scheduler, vae=pipe.vae, text_encoder=None,
                                        tokenizer=pipe.tokenizer, transformer=pipe.transformer, is_distilled=True)
            drawer.set_progress_bar_config(disable=True)
            drawer.__class__ = type("Drawer", (Flux2KleinPipeline,),
                                    {"_execution_device": property(lambda _self: self.device)})
            return drawer(image=refs or None, prompt_embeds=embeds, width=w, height=h, num_inference_steps=int(steps),
                          guidance_scale=1.0, generator=torch.Generator(self.device).manual_seed(int(seed))).images[0]

    def __call__(self, picture, mask, swatch, prompt="", steps=6, strength=1.0, seed=0, focus=True):
        """`picture` (PIL) redrawn inside `mask` ((h, w) in [0, 1]) after the `swatch` (PIL). With `focus`,
        only the mask's surroundings are worked on and laid back into the picture."""
        box = around(mask) if focus else None
        if box is not None:
            whole, (x0, y0, x1, y1), (ww, hh) = picture, box, worksize(box, picture.size)
            picture = picture.crop(box).resize((ww, hh), Image.LANCZOS)
            mask = resized(mask[y0:y1, x0:x1], ww, hh)
        out = self.draw(picture, mask, swatch, prompt, steps, strength, seed)
        if box is None:
            return out
        whole = whole.copy()
        whole.paste(out.resize((x1 - x0, y1 - y0), Image.LANCZOS), (x0, y0))
        return whole

    def draw(self, picture, mask, swatch, prompt, steps, strength, seed):
        import torch

        w, h = picture.size
        if w * h > 1024 * 1024:
            k = math.sqrt(1024 * 1024 / (w * h))
            w, h = int(w * k) // 16 * 16, int(h * k) // 16 * 16
            picture = picture.resize((w, h), Image.LANCZOS)
            mask = resized(mask, w, h)
        else:
            w, h = w // 16 * 16, h // 16 * 16
            picture = picture.resize((w, h), Image.LANCZOS)
            mask = resized(mask, w, h)
        mask_im = Image.fromarray((np.clip(mask, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8), "L")
        swatch = swatch.convert("RGB")
        swatch.thumbnail((SWATCH_SIDE, SWATCH_SIDE), Image.LANCZOS)
        with self.lock, torch.no_grad():
            pipe = self.load()
            words = prompt or instruction(None)
            embeds = self.words(pipe, words, 128 if len(words.split()) <= 80 else 512)
            pipe = self.load()                     # the image model back, if the words took its place
            return pipe(prompt_embeds=embeds, image=picture.convert("RGB"), mask_image=mask_im,
                        image_reference=swatch, width=w, height=h, strength=float(strength),
                        num_inference_steps=int(steps), guidance_scale=1.0,
                        generator=torch.Generator(self.device).manual_seed(int(seed))).images[0]


def _read(handle, key, tries=4):
    """One tensor from an open safetensors file. Under memory pressure the read can fail inside the
    memory map ("invalid python storage"); a moment later, with the cache collected, it goes through."""
    import gc
    import time

    for attempt in range(tries):
        try:
            return handle.get_tensor(key)
        except RuntimeError:
            if attempt == tries - 1:
                raise
            gc.collect()
            time.sleep(1.0 + attempt)


def to_card(cls, subfolder: str):
    """A model built empty and filled from its safetensors one tensor at a time, each read straight onto
    the card. diffusers and transformers read a whole file through RAM first (8 GB here), and on a
    machine whose RAM and page file are nearly committed that read dies inside the memory map with an
    access violation; this one never holds more than a tensor in RAM."""
    import glob
    import os

    import torch
    from accelerate import init_empty_weights
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    root = os.path.join(snapshot_download(KLEIN, local_files_only=True), subfolder)
    if hasattr(cls, "load_config"):                   # a diffusers model
        cfg = cls.load_config(root)
        with init_empty_weights():
            model = cls.from_config(cfg)
    else:                                             # a transformers model
        from transformers import AutoConfig

        cfg = _hub.pretrained(AutoConfig.from_pretrained, root)
        with init_empty_weights():
            model = cls(cfg)
    state = {}
    for f in sorted(glob.glob(os.path.join(root, "*.safetensors"))):
        with safe_open(f, framework="pt", device="cuda") as h:
            for k in h.keys():
                state[k] = _read(h, k).to(torch.bfloat16)
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    left = [k for k in missing if "lm_head" not in k]
    if left or unexpected:
        raise RuntimeError(f"{subfolder}: missing {left[:4]}, unexpected {list(unexpected)[:4]}")
    for name, buf in list(model.named_buffers()):
        if buf.device.type == "meta":
            raise RuntimeError(f"{subfolder}: buffer {name} was never filled")
    return model.to("cuda").eval()


EDITOR = Editor()


def park():
    """Make room on the card for Stable Diffusion: the editor's image model goes to RAM."""
    EDITOR.park()


def _free_diffusion():
    """Stable Diffusion's pipelines off the card before the editor loads: both do not fit at once."""
    try:
        from .. import style_transfer as st
        from .. import ip_labels
    except ImportError:                               # pragma: no cover
        import style_transfer as st
        import ip_labels
    st._PIPE_CACHE.clear()
    st._CN_CACHE.clear()
    st._T2I_CACHE.clear()
    ip_labels.release_encoder()
    import gc
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

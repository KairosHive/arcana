"""DINOv2's tokens and the decoder that draws them back (RAE).

A picture becomes a grid of DINOv2-B-with-registers tokens, one per 16 x 16 pixels, and a ViT decoder
trained by Zheng, Ma, Tong and Xie ("Diffusion Transformers with Representation Autoencoders", 2025,
MIT) draws a picture back from any grid. Two materials mixed in that grid stay one surface, which is
what makes token mixing different from a crossfade. Ported from latent-lab's rae.py.
"""
try:
    from arcana import hub as _hub
except ImportError:                                   # pragma: no cover
    import hub as _hub
import math
import pathlib

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

ENCODER = "facebook/dinov2-with-registers-base"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
SIZES = {768: dict(heads=12, layers=12, inner=3072), 1152: dict(heads=16, layers=28, inner=4096)}  # ViT-B, ViT-XL


def load_mapped(path):
    """A checkpoint read memory-mapped: its tensors are paged in from the file rather than copied into
    the process's own memory first, so a load does not briefly commit the whole file twice (the
    lab runs close to its commit limit). Falls back to a plain read for a file that cannot be mapped."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except RuntimeError:
        return torch.load(path, map_location="cpu", weights_only=True)


class Holder(nn.Module):
    """A named slot, so the attribute paths match the checkpoint's keys."""

    def __init__(self, **parts):
        super().__init__()
        for name, part in parts.items():
            setattr(self, name, part)


class Layer(nn.Module):
    """ViT-MAE's block: pre-norm attention, pre-norm GELU MLP, both residual."""

    def __init__(self, width, heads, inner):
        super().__init__()
        self.heads = heads
        self.attention = Holder(attention=Holder(query=nn.Linear(width, width), key=nn.Linear(width, width),
                                                 value=nn.Linear(width, width)),
                                output=Holder(dense=nn.Linear(width, width)))
        self.intermediate = Holder(dense=nn.Linear(width, inner))
        self.output = Holder(dense=nn.Linear(inner, width))
        self.layernorm_before = nn.LayerNorm(width, eps=1e-12)
        self.layernorm_after = nn.LayerNorm(width, eps=1e-12)

    def forward(self, x):
        b, n, c = x.shape
        att = self.attention.attention
        h = self.layernorm_before(x)
        q, k, v = (proj(h).view(b, n, self.heads, c // self.heads).transpose(1, 2) for proj in (att.query, att.key, att.value))
        y = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, c)
        x = x + self.attention.output.dense(y)
        return x + self.output.dense(F.gelu(self.intermediate.dense(self.layernorm_after(x))))


class Decoder(nn.Module):
    def __init__(self, patches, width, tokens=768, patch=16):
        super().__init__()
        cfg = SIZES[width]
        self.patch, self.side = patch, int(math.isqrt(patches))
        self.decoder_embed = nn.Linear(tokens, width)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, patches + 1, width), requires_grad=False)
        self.trainable_cls_token = nn.Parameter(torch.zeros(1, 1, width))
        self.decoder_layers = nn.ModuleList(Layer(width, cfg["heads"], cfg["inner"]) for _ in range(cfg["layers"]))
        self.decoder_norm = nn.LayerNorm(width, eps=1e-12)
        self.decoder_pred = nn.Linear(width, patch * patch * 3)

    def positions(self, h, w):
        """The position table for an h x w grid no larger than the trained square: its top-left corner."""
        if (h, w) == (self.side, self.side):
            return self.decoder_pos_embed
        pe = self.decoder_pos_embed
        corner = pe[:, 1:].reshape(1, self.side, self.side, -1)[:, :h, :w].reshape(1, h * w, -1)
        return torch.cat([pe[:, :1], corner], 1)

    def forward(self, z, grid=None):
        """[b, h*w, 768] tokens -> [b, 3, h*16, w*16] pixels, ImageNet-normalised; the grid is square unless
        given, and no larger than the trained one (RAE.decode draws larger ones in windows)."""
        h, w = grid or (self.side, self.side)
        x = self.decoder_embed(z)
        x = torch.cat([self.trainable_cls_token.expand(len(x), -1, -1), x], 1) + self.positions(h, w)
        for layer in self.decoder_layers:
            x = layer(x)
        x = self.decoder_pred(self.decoder_norm(x))[:, 1:]
        p = self.patch
        return torch.einsum("nhwpqc->nchpwq", x.reshape(len(x), h, w, p, p, 3)).reshape(len(x), 3, h * p, w * p)


class RAE:
    NORMALISED = True  # the decoder draws ImageNet-normalised pixels (RAEv2's draws 0..1)

    def __init__(self, decoder, device="cuda", half=True):
        self.path = str(decoder)
        state = load_mapped(decoder)
        for wrap in ("model", "ema", "state_dict"):
            if isinstance(state, dict) and wrap in state and isinstance(state[wrap], dict):
                state = state[wrap]
        state = {k.removeprefix("decoder."): v for k, v in state.items()}
        patches, width = state["decoder_pos_embed"].shape[1] - 1, state["decoder_pos_embed"].shape[2]
        self.decoder = Decoder(patches, width, state["decoder_embed.weight"].shape[1])
        missing, unexpected = self.decoder.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise SystemExit(f"{decoder}: missing {missing[:4]}, unexpected {unexpected[:4]}")
        self.side = self.decoder.side
        self.size = self.side * 16
        self.dtype = torch.float16 if half and device != "cpu" else torch.float32
        self.device = device
        self.encoder = self.load_encoder().to(device, getattr(self, "encoder_dtype", self.dtype)).eval()
        self.decoder.to(device, self.dtype).eval()
        if device != "cpu":
            torch.backends.cudnn.benchmark = True

    def load_encoder(self):
        """DINOv2-B with registers, its last layer norm without its learned scale and shift."""
        from transformers import Dinov2WithRegistersModel

        encoder = _hub.pretrained(Dinov2WithRegistersModel.from_pretrained, ENCODER, attn_implementation="sdpa")
        encoder.layernorm.elementwise_affine = False
        encoder.layernorm.weight = None
        encoder.layernorm.bias = None
        return encoder

    def shell(self, z):
        """A mix of tokens put back where real tokens lie: for DINOv2, `renorm`."""
        return renorm(z)

    def grid(self, width, height, long=None):
        """The token grid (rows, columns) for a frame of this shape, `long` tokens on its long side."""
        long = long or self.side
        if width >= height:
            return max(1, round(long * height / width)), long
        return long, max(1, round(long * width / height))

    @torch.no_grad()
    def encode(self, images, grid=None):
        """[b, h, w, 3] or [h, w, 3] in 0..1 -> [b, rows*cols, 768] tokens (the picture fitted to the grid,
        square unless one is given)."""
        rows, cols = grid or (self.side, self.side)
        x = torch.as_tensor(np.asarray(images), dtype=torch.float32)
        x = (x[None] if x.ndim == 3 else x).permute(0, 3, 1, 2).to(self.device)
        x = F.interpolate(x, size=(rows * 14, cols * 14), mode="bicubic", align_corners=False)
        x = ((x - MEAN.to(x.device)) / STD.to(x.device)).to(self.dtype)
        return self.encoder(pixel_values=x).last_hidden_state[:, 5:]  # 1 class + 4 register tokens first

    def spans(self, n, overlap=8):
        """Where the windows start along one side of a grid: one if the side fits the trained table, else
        as few as cover it with at least `overlap` tokens shared between neighbours."""
        if n <= self.side:
            return [0]
        k = math.ceil((n - self.side) / (self.side - overlap)) + 1
        return [round(i * (n - self.side) / (k - 1)) for i in range(k)]

    @torch.no_grad()
    def decode(self, tokens, grid=None):
        """[b, rows*cols, 768] -> [b, rows*16, cols*16, 3] in 0..1 (float32, on the CPU)."""
        rows, cols = grid or (self.side, self.side)
        z = tokens.to(self.device, self.dtype)
        ys, xs = self.spans(rows), self.spans(cols)
        hh, ww = min(rows, self.side), min(cols, self.side)
        g = z.view(len(z), rows, cols, -1)
        pieces = torch.cat([g[:, y:y + hh, x:x + ww].reshape(len(z), hh * ww, -1) for y in ys for x in xs])
        x = self.decoder(pieces, (hh, ww)).float()
        if self.NORMALISED:
            x = x * STD.to(x.device) + MEAN.to(x.device)
        x = x.clamp(0, 1)
        if len(ys) * len(xs) == 1:
            return x.permute(0, 2, 3, 1).cpu().numpy()
        # Each window weighs most in its middle and fades to almost nothing at its edges, over the
        # overlap, so neighbours cross-fade; at the frame's own edge one window is all there is.
        ramp = lambda n: torch.clamp(torch.minimum(torch.arange(n) + 0.5, n - torch.arange(n) - 0.5) / 128, 0.01, 1)  # noqa: E731
        weight = (ramp(hh * 16)[:, None] * ramp(ww * 16)[None]).to(x.device)
        out = torch.zeros(len(z), 3, rows * 16, cols * 16, device=x.device)
        total = torch.zeros(1, 1, rows * 16, cols * 16, device=x.device)
        x = x.view(len(ys) * len(xs), len(z), 3, hh * 16, ww * 16)
        for k, (y, xx) in enumerate((y, xx) for y in ys for xx in xs):
            out[:, :, y * 16:(y + hh) * 16, xx * 16:(xx + ww) * 16] += x[k] * weight
            total[:, :, y * 16:(y + hh) * 16, xx * 16:(xx + ww) * 16] += weight
        return (out / total).permute(0, 2, 3, 1).cpu().numpy()


def renorm(z):
    """Tokens back onto the encoder's shell (each zero-mean, unit spread over its 768 numbers, as its
    last layer norm leaves them): a mix of two tokens is shorter than either."""
    return F.layer_norm(z.float(), (z.shape[-1],)).to(z.dtype)


def along(grids, u):
    """The closed Catmull-Rom curve through a loop of token grids, at u in [0, 1): through every one,
    stopping at none."""
    n = len(grids)
    x = float(u) % 1.0 * n
    k, s = int(x), x - int(x)
    p0, p1, p2, p3 = (grids[(k + d) % n].float() for d in (-1, 0, 1, 2))
    return 0.5 * (2 * p1 + (p2 - p0) * s + (2 * p0 - 5 * p1 + 4 * p2 - p3) * s * s + (3 * p1 - p0 - 3 * p2 + p3) * s ** 3)


def piece(path, width, height, zoom=1.0):
    """The middle of a photograph, `1 / zoom` of it across, in a frame of this shape and size (a large
    JPEG decoded at reduced scale)."""
    with Image.open(path) as im:
        im.draft("RGB", (width * 2, height * 2))
        im = im.convert("RGB")
    w, h = im.size
    cw, ch = (w, round(w * height / width)) if w * height <= h * width else (round(h * width / height), h)
    cw, ch = round(cw / zoom), round(ch / zoom)
    left, top = (w - cw) // 2, (h - ch) // 2
    return np.asarray(im.crop((left, top, left + cw, top + ch)).resize((width, height), Image.LANCZOS), np.float32) / 255


def load(decoder, device="cuda"):
    """The autoencoder a decoder belongs to: one trained on DINOv3's tokens (a path naming dinov3, as
    RAEv2's do) is raev2.py's, anything else DINOv2's."""
    return RAE(decoder, device)



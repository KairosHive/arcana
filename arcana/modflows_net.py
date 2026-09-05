# modflows_net.py — colour transfer by matched rectified flows
#
# Our own implementation of the inference path for the ModFlows colour encoder
# (Larchenko et al., AAAI 2025, arXiv 2503.19062).
#
# WHY THIS EXISTS. The published checkpoint is MIT
# (MariaLarchenko/modflows_color_encoder on HuggingFace) but the reference
# implementation at github.com/maria-larchenko/modflows carries NO LICENSE at
# all, so default copyright applies and it cannot be redistributed. Arcana's
# installer was bundling five files from it. This module replaces them, so the
# packaged app ships only code we own alongside weights we are licensed to
# fetch. See LICENSING.md, item A0.
#
# HOW IT WORKS. Each image gets a rectified flow that carries its own colour
# distribution to a shared latent one. Transfer is then two integrations: push
# the content pixels FORWARD along the content image's flow into the latent
# space, then pull them BACKWARD along the style image's flow. Colours that
# occupied a position in the content's distribution come out at the
# corresponding position in the style's.
#
# One EfficientNet-B6 predicts, from a single image, all 8,195 parameters of the
# tiny MLP that defines that image's velocity field:
#
#     4*1024 (W1) + 1024 (b1) + 3*1024 (W2) + 3 (b2) = 8195
#
# The MLP maps (r, g, b, t) to a velocity in RGB. Four inputs, three outputs,
# one hidden layer of 1024 with tanh.

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image

ENCODER_INPUT = 528          # what the B6 checkpoint was trained at
K_DIM = 8195
INPUT_DIM = 4                # r, g, b, t
HIDDEN = 1024
OUTPUT_DIM = 3               # velocity in r, g, b


class ColorFlowEncoder(nn.Module):
    """
    EfficientNet-B6 whose classifier emits the parameters of a velocity field.

    The submodule is named `model` and nothing is added around it, because the
    checkpoint is a state_dict keyed `model.features.*` and `model.classifier.*`
    -- torchvision's own names. Any other structure would fail to load, which is
    what `strict=True` in from_checkpoint() is there to prove.
    """

    def __init__(self, k_dim: int = K_DIM, arch: str = "B6"):
        super().__init__()
        if arch == "B6":
            self.model = torchvision.models.efficientnet_b6(num_classes=k_dim)
            weights = torchvision.models.EfficientNet_B6_Weights.IMAGENET1K_V1
        elif arch == "B0":
            self.model = torchvision.models.efficientnet_b0(num_classes=k_dim)
            weights = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1
        else:
            raise ValueError(f"unknown encoder architecture: {arch!r}")
        # The same normalisation the backbone was pretrained under. Applied
        # inside forward(), as it was during training.
        self.normalise = weights.transforms()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.normalise(x)
        return self.model(x)

    @classmethod
    def from_checkpoint(cls, path, device, k_dim: int = K_DIM, arch: str = "B6"):
        enc = cls(k_dim=k_dim, arch=arch)
        state = torch.load(str(path), map_location="cpu", weights_only=True)
        # strict: a silently partial load would produce a random velocity field
        # and a plausible-looking wrong picture, which is the worst failure this
        # module could have.
        enc.load_state_dict(state, strict=True)
        enc.eval()
        return enc.to(device)


def preprocess(pil_image: Image.Image, device) -> torch.Tensor:
    """
    Turn a picture into the encoder's input tensor.

    NOTE the reshape. The reference implementation resizes to 528x528, scales to
    [0,1], and then calls .reshape(3, 528, 528) on an array whose shape is
    (528, 528, 3). That REINTERPRETS the buffer rather than transposing it, so
    the tensor handed to the network is not the image in CHW order -- it is the
    same bytes read in a different order.

    Almost certainly not what was intended, but the encoder was trained through
    it, so its weights only mean anything on inputs prepared the same way.
    Transposing properly here produces a different, wrong velocity field.
    Reproduced deliberately; do not "fix".
    """
    # torchvision's Resize, not PIL's, because that is what the encoder saw
    # during training. They differ enough at this scale to move the predicted
    # velocity field, and the whole point of matching the checkpoint is to feed
    # it what it expects.
    from torchvision.transforms import v2 as _v2
    im = _v2.Resize((ENCODER_INPUT, ENCODER_INPUT))(pil_image.convert("RGB"))
    arr = np.asarray(im, dtype=np.float32) / 255.0
    arr = arr.reshape(3, ENCODER_INPUT, ENCODER_INPUT)      # see above
    return torch.from_numpy(arr).unsqueeze(0).to(device)


def split_params(e: torch.Tensor):
    """
    Cut the encoder's 8,195 outputs into the velocity field's four tensors.

    Returned in the shapes nn.Linear would hold them: W1 (hidden, in),
    b1 (hidden), W2 (out, hidden), b2 (out).
    """
    a = INPUT_DIM * HIDDEN
    b = a + HIDDEN
    c = b + OUTPUT_DIM * HIDDEN
    d = c + OUTPUT_DIM
    if e.numel() != d:
        raise ValueError(f"expected {d} parameters, got {e.numel()}")
    e = e.flatten()
    return (e[0:a].reshape(HIDDEN, INPUT_DIM),
            e[a:b].reshape(HIDDEN),
            e[b:c].reshape(OUTPUT_DIM, HIDDEN),
            e[c:d].reshape(OUTPUT_DIM))


def velocity(params, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    The field itself: (pixels, time) -> a direction to move each pixel.

    x is (N, 3) of RGB in [0,1]; t is (N, 1). Two matrix multiplies with a tanh
    between them -- small enough that the cost of a transfer is the number of
    integration steps, not the width of this.
    """
    w1, b1, w2, b2 = params
    h = torch.cat([x, t], dim=-1) @ w1.T + b1
    return torch.tanh(h) @ w2.T + b2


PIXEL_CHUNK = 262_144


@torch.no_grad()
def integrate(params, x: torch.Tensor, steps: int, strength: float,
              reverse: bool = False, chunk: int = PIXEL_CHUNK) -> torch.Tensor:
    """
    Walk the pixels along the flow with a forward Euler solver.

    `strength` stops the walk early rather than scaling the field, so a partial
    transfer is a partial journey along the same path -- the colours land
    somewhere the flow actually passes through, instead of somewhere a scaled
    velocity would have invented.
    """
    dt = 1.0 / steps
    stop = int(strength * steps)
    out = torch.empty_like(x)

    # Every pixel passes through a 1024-wide hidden layer, so a megapixel needs
    # about a gigabyte for that intermediate alone -- enough to run a GPU out of
    # memory on a large image. Each pixel's path is independent of every
    # other's, so chunking changes nothing about the result.
    for start in range(0, x.shape[0], chunk):
        z = x[start:start + chunk].clone()
        ones = torch.ones((z.shape[0], 1), device=z.device, dtype=z.dtype)
        for i in range(steps):
            t = ones * (i / steps)
            z = z - velocity(params, z, 1.0 - t) * dt if reverse \
                else z + velocity(params, z, t) * dt
            if i > stop:
                break
        out[start:start + chunk] = z
    return out


@torch.no_grad()
def transfer(encoder: ColorFlowEncoder, content: Image.Image, style: Image.Image,
             device, steps: int = 8, strength: float = 1.0,
             max_side: int | None = None) -> Image.Image:
    """
    Give `content` the colours of `style`.

    Both images are encoded to their own flow. The content's pixels are pushed
    forward along its flow into the shared latent distribution, then pulled back
    along the style's -- so a colour sitting at some position in the content's
    distribution emerges at the matching position in the style's.

    Only the CONTENT is resized by `max_side`: the flow runs per pixel, so cost
    scales with how many there are. The style image never contributes pixels,
    only the 8,195 numbers describing its distribution.
    """
    content = content.convert("RGB")
    style = style.convert("RGB")

    c_params = split_params(encoder(preprocess(content, device)).flatten())
    s_params = split_params(encoder(preprocess(style, device)).flatten())

    work = content
    if max_side and max(content.size) > max_side:
        scale = max_side / float(max(content.size))
        # PIL's own default resampler, matching the reference. LANCZOS would be
        # a defensible choice on its own, but it shifts pixel values enough to
        # make "does this still agree with what it replaced" untestable.
        work = content.resize((max(1, int(content.width * scale)),
                               max(1, int(content.height * scale))))

    w, h = work.size
    px = np.asarray(work, dtype=np.float32).reshape(w * h, 3) / 255.0
    x = torch.from_numpy(px).to(device)

    latent = integrate(c_params, x, steps, strength, reverse=False)
    styled = integrate(s_params, latent, steps, strength, reverse=True)

    out = torch.clip(styled, 0.0, 1.0).reshape(h, w, 3).cpu().numpy()
    return Image.fromarray((out * 255).astype(np.uint8))

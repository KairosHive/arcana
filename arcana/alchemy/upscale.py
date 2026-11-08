"""Real-ESRGAN x4 (xinntao, BSD-3): a picture four times larger, the weights read as tensors only.
Ported from latent-lab's upscale.py.
"""
import pathlib

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn


class Dense(nn.Module):
    def __init__(self, feat=64, grow=32):
        super().__init__()
        self.conv1 = nn.Conv2d(feat, grow, 3, 1, 1)
        self.conv2 = nn.Conv2d(feat + grow, grow, 3, 1, 1)
        self.conv3 = nn.Conv2d(feat + 2 * grow, grow, 3, 1, 1)
        self.conv4 = nn.Conv2d(feat + 3 * grow, grow, 3, 1, 1)
        self.conv5 = nn.Conv2d(feat + 4 * grow, feat, 3, 1, 1)

    def forward(self, x):
        seen = [x]
        for conv in (self.conv1, self.conv2, self.conv3, self.conv4):
            seen.append(F.leaky_relu(conv(torch.cat(seen, 1)), 0.2))
        return self.conv5(torch.cat(seen, 1)) * 0.2 + x


class Block(nn.Module):
    def __init__(self, feat=64, grow=32):
        super().__init__()
        self.rdb1, self.rdb2, self.rdb3 = Dense(feat, grow), Dense(feat, grow), Dense(feat, grow)

    def forward(self, x):
        return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x


class Net(nn.Module):
    """basicsr's RRDBNet at scale 4, under the names its checkpoints use."""

    def __init__(self, feat=64, blocks=23, grow=32):
        super().__init__()
        self.conv_first = nn.Conv2d(3, feat, 3, 1, 1)
        self.body = nn.Sequential(*[Block(feat, grow) for _ in range(blocks)])
        self.conv_body = nn.Conv2d(feat, feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(feat, feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(feat, feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(feat, feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(feat, 3, 3, 1, 1)

    def forward(self, x):
        feat = self.conv_first(x)
        feat = feat + self.conv_body(self.body(feat))
        feat = F.leaky_relu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")), 0.2)
        feat = F.leaky_relu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")), 0.2)
        return self.conv_last(F.leaky_relu(self.conv_hr(feat), 0.2))


def load_mapped(path):
    """A checkpoint read memory-mapped: its tensors are paged in from the file rather than copied into
    the process's own memory first, so a load does not briefly commit the whole file twice (the
    lab runs close to its commit limit). Falls back to a plain read for a file that cannot be mapped."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except RuntimeError:
        return torch.load(path, map_location="cpu", weights_only=True)


def load(weights, device="cuda"):
    state = load_mapped(weights)
    net = Net()
    net.load_state_dict(state.get("params_ema", state.get("params", state)))
    net = net.eval().to(device)
    if device == "cuda":
        torch.backends.cudnn.benchmark = True  # a film is thousands of frames of one size
        return net.half().to(memory_format=torch.channels_last)  # a fifth faster on a 3090
    return net


@torch.no_grad()
def enlarge(net, image, keep=False):
    """[h, w, 3] floats in 0..1 to [4h, 4w, 3]. `keep`: left on the card as a float tensor, for a
    caller that finishes the picture there and copies back bytes rather than floats (a 3072 x 2048
    float frame is 75 MB to bring back, its bytes a quarter of that)."""
    p = next(net.parameters())
    x = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))[None].to(p.device, p.dtype)
    # A few pixels of mirror round the edge, so the border is drawn with neighbours like the rest.
    pad = 8
    big = net(F.pad(x, (pad, pad, pad, pad), mode="reflect"))[..., 4 * pad : -4 * pad, 4 * pad : -4 * pad]
    big = big.float().clamp(0, 1)[0].permute(1, 2, 0)
    return big if keep else big.cpu().numpy()



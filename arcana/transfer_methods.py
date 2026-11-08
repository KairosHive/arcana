# transfer_methods.py -- the colour transfer methods, as one table
#
# Before this, the method names were spelled out in the app: the dropdown, then
# a `method == "lab"` branch for which controls to grey out, another for the
# quality note, another for availability, another for dispatch and another for
# the filename. An agent would have been a seventh place. Now the table below
# is the only place a method is described, and transfer() the only dispatch,
# following models.py, which is the house pattern for a registry.
#
# Ordered cheapest first. Timings are measured, not guessed: one 1024 px
# content picture and one reference, the whole call including loading, on the
# machine named beside them. See tests/test_transfer_methods.py for what each
# method must do.

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

from PIL import Image


@dataclass(frozen=True)
class TransferMethod:
    id: str
    label: str              # what a person sees
    licence: str            # of the method's code and weights as shipped
    needs_download_mb: int  # 0 means nothing to fetch
    needs_gpu: bool         # False means it runs on the CPU build
    cpu_ms: float           # measured at 1024 px
    gpu_ms: float
    quality: str
    params: tuple[str, ...] # which controls apply
    blurb: str

    def as_dict(self) -> dict:
        return asdict(self)

    def timing(self, has_gpu: bool) -> str:
        ms = self.gpu_ms if has_gpu else self.cpu_ms
        if ms < 1000:
            return "under a second"
        return f"about {ms / 1000:.0f} s"


# Measured 7 November 2026: 1024 x 683 content, the median of three warm runs,
# on an Intel Core Ultra 9 185H (CPU) and an RTX 4090 Laptop GPU. ModFlows at
# its balanced preset. The first call in a process costs more: about 8 s for
# IDT (scipy's import) and 2 s for ModFlows on the GPU (loading the model).
METHODS: tuple[TransferMethod, ...] = (
    TransferMethod(
        id="lab", label="LAB (Reinhard)", licence="own code", needs_download_mb=0,
        needs_gpu=False, cpu_ms=50, gpu_ms=50, quality="quick",
        params=("strength",),
        blurb="Matches each channel's mean and spread in LAB. Instant and safe for a "
              "small nudge. It can leave a colour cast, since it treats the channels "
              "one at a time."),
    TransferMethod(
        id="mkl", label="MKL (linear)", licence="own code", needs_download_mb=0,
        needs_gpu=False, cpu_ms=70, gpu_ms=70, quality="good",
        params=("strength",),
        blurb="Maps the whole colour covariance onto the reference's, the best linear "
              "transfer there is. Instant. Fixes the casts LAB leaves."),
    TransferMethod(
        id="idt", label="IDT (distribution)", licence="own code", needs_download_mb=0,
        needs_gpu=False, cpu_ms=1000, gpu_ms=1000, quality="better",
        params=("strength",),
        blurb="Matches the full 3-D colour distribution by repeated 1-D matches along "
              "random axes. Takes on the reference's palette more completely than MKL, "
              "in about a second."),
    TransferMethod(
        id="modflows", label="ModFlows (neural)", licence="MIT code, CC-BY-NC weights",
        needs_download_mb=229, needs_gpu=False, cpu_ms=3400, gpu_ms=470, quality="best",
        params=("strength", "quality"),
        blurb="A learned colour flow. The most faithful to the reference's mood, and the "
              "slowest on a CPU. Needs a one-off 229 MB model."),
)

BY_ID = {m.id: m for m in METHODS}
QUALITIES = ("quick", "balanced", "best")


def get(method_id: str) -> TransferMethod | None:
    return BY_ID.get(method_id)


def available(method_id: str) -> tuple[bool, str]:
    """(can it run here, and if not, why) for one method."""
    m = get(method_id)
    if m is None:
        return False, f"no method called {method_id!r}. Methods: {', '.join(BY_ID)}."
    if method_id == "modflows":
        try:
            from . import color_transfer as _ct
        except ImportError:                           # pragma: no cover
            import color_transfer as _ct
        try:
            import torch  # noqa: F401
        except ImportError:
            return False, "ModFlows needs PyTorch, which this build lacks."
        if not _ct.status()["ready"]:
            return False, ("ModFlows needs its 229 MB model first. Fetch it with "
                           "`model download modflows`.")
    return True, ""


def default_for(has_gpu: bool) -> TransferMethod:
    """ModFlows when it can run, MKL otherwise."""
    ok, _ = available("modflows")
    return BY_ID["modflows"] if ok else BY_ID["mkl"]


def transfer(method_id: str, content: Image.Image, reference: Image.Image,
             strength: float = 1.0, quality: str = "balanced",
             content_path: str | None = None) -> tuple[Image.Image, dict]:
    """
    Recolour `content` toward `reference` with one method. Returns the picture
    at the content's full size and what ran: {"label", "device", "elapsed"}.
    """
    ok, why = available(method_id)
    if not ok:
        raise ValueError(why)
    started = time.time()
    content = content.convert("RGB")
    reference = reference.convert("RGB")
    device = "CPU"
    if method_id == "lab":
        try:
            from .lab_transfer import lab_color_transfer_pil
        except ImportError:                           # pragma: no cover
            from lab_transfer import lab_color_transfer_pil
        out = lab_color_transfer_pil(content, reference, strength=strength)
        label = BY_ID["lab"].label
    elif method_id in ("mkl", "idt"):
        try:
            from . import classic_transfer as _ctr
        except ImportError:                           # pragma: no cover
            import classic_transfer as _ctr
        out = (_ctr.mkl if method_id == "mkl" else _ctr.idt)(content, reference, strength)
        label = BY_ID[method_id].label
    else:
        try:
            from . import color_transfer as _ct
        except ImportError:                           # pragma: no cover
            import color_transfer as _ct
        if quality not in QUALITIES:
            raise ValueError(f"quality is one of {', '.join(QUALITIES)}, got {quality!r}.")
        preset = _ct.QUALITY_PRESETS[quality]
        low = _ct.transfer_colors(content=content, style=reference, strength=strength,
                                  steps=preset["steps"], max_size=preset["max_size"])
        if content_path:
            out = _ct.transfer_at_full_resolution(content_path, low, preset["lut"])
        else:
            small = content.resize(low.size, Image.LANCZOS)
            out = _ct._apply_lut3d(content, _ct._fit_lut3d(small, low, preset["lut"]))
        label = f"ModFlows, {preset['label'].lower()}"
        device = (_ct.get_device_info() or {}).get("device", "cpu").upper()
    return out, {"label": label, "device": device, "elapsed": time.time() - started}

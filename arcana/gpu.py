# gpu.py -- one honest answer to "can this machine actually use its GPU?"
#
# torch.cuda.is_available() is not that answer. It reports whether a CUDA
# driver and device are present, not whether the torch build you are holding
# can run anything on them. The wheel Arcana ships is compiled for a fixed set
# of architectures:
#
#     torch 2.9.1+cu128 -> sm_70 75 80 86 90 100 120
#
# There is no sm_6x in that list. On a GTX 1080 -- Pascal, sm_61 -- CUDA is
# "available", every probe in the app says yes, the encoder chooser promotes
# the user to ViT-H/14 on the strength of it, they download 3,945 MB of
# weights, and then the first forward pass dies with "no kernel image is
# available for execution on the device", part-way through indexing, into a log
# file nobody is watching. Nothing before that moment is a warning.
#
# So the probe here does three things instead of one: it checks that the
# device's compute capability is in the build's arch list, it runs a real (if
# tiny) kernel to catch everything the arch list cannot predict -- an old
# driver, an exhausted card, a laptop GPU switched off by the OS -- and it
# caches the verdict so nine call sites do not each pay for it.
#
# Everything here is deliberately unable to raise. A capability probe that
# throws is worse than one that returns False, because it turns "no GPU" into
# a crash on a machine that was working fine.

from __future__ import annotations

import os

# Filled on first use: {"ok": bool, "device": str, "reason": str, "name": str,
#                       "capability": str, "arch_list": list[str]}
_VERDICT: dict | None = None

ENV_FORCE_CPU = "ARCANA_FORCE_CPU"

# Volta (sm_70) is where torch's fp16 tensor-core path becomes reliable.
# Below it, .half() runs but is slow and numerically worse than fp32, so the
# encoder should stay in fp32 even where CUDA works.
_FP16_MIN_MAJOR = 7


def _probe() -> dict:
    """Work out, once, what this machine can really do."""
    bad = {"ok": False, "device": "cpu", "name": "", "capability": "",
           "arch_list": [], "fp16": False}

    if os.environ.get(ENV_FORCE_CPU):
        return {**bad, "reason": f"{ENV_FORCE_CPU} is set"}

    try:
        import torch
    except Exception as e:                                   # pragma: no cover
        return {**bad, "reason": f"torch did not import ({type(e).__name__})"}

    try:
        if not torch.cuda.is_available():
            return {**bad, "reason": "no CUDA device visible to torch"}
    except Exception as e:
        # A broken or half-installed driver can make even this raise.
        return {**bad, "reason": f"CUDA probe failed ({type(e).__name__})"}

    try:
        name = torch.cuda.get_device_name(0)
        major, minor = torch.cuda.get_device_capability(0)
        arch_list = list(torch.cuda.get_arch_list())
    except Exception as e:
        return {**bad, "reason": f"could not query the device ({type(e).__name__})"}

    cap = f"sm_{major}{minor}"
    # torch can also run a newer card via a lower sm_ target, so accept any
    # compiled arch at or below this device's capability rather than requiring
    # an exact string match.
    def _n(a: str) -> int:
        try:
            return int(a.split("_", 1)[1])
        except Exception:
            return 10_000
    usable = [a for a in arch_list if a.startswith("sm_") and _n(a) <= major * 10 + minor]
    if not usable:
        return {**bad, "name": name, "capability": cap, "arch_list": arch_list,
                "reason": (f"{name} is {cap}, and this build of torch only has "
                           f"kernels for {', '.join(arch_list) or 'nothing'}")}

    # The arch list cannot see an old driver, a card already out of memory, or
    # a laptop GPU the OS has powered down. One real kernel can.
    try:
        a = torch.zeros((8, 8), device="cuda")
        b = (a + 1.0) @ (a + 1.0)
        torch.cuda.synchronize()
        if not bool(torch.isfinite(b).all()):
            raise RuntimeError("test kernel produced non-finite values")
    except Exception as e:
        return {**bad, "name": name, "capability": cap, "arch_list": arch_list,
                "reason": f"a test kernel failed on {name} ({type(e).__name__}: {e})"}

    return {"ok": True, "device": "cuda", "name": name, "capability": cap,
            "arch_list": arch_list, "fp16": major >= _FP16_MIN_MAJOR,
            "reason": f"{name} ({cap})"}


def verdict(refresh: bool = False) -> dict:
    """The cached capability verdict. Never raises."""
    global _VERDICT
    if _VERDICT is None or refresh:
        try:
            _VERDICT = _probe()
        except Exception as e:                               # pragma: no cover
            _VERDICT = {"ok": False, "device": "cpu", "name": "", "capability": "",
                        "arch_list": [], "fp16": False,
                        "reason": f"capability probe crashed ({type(e).__name__})"}
        if not _VERDICT["ok"] and _VERDICT.get("reason"):
            print(f"[gpu] using the CPU: {_VERDICT['reason']}")
    return _VERDICT


def available() -> bool:
    """True only if a GPU is present AND this torch build can actually use it."""
    return bool(verdict()["ok"])


def device() -> str:
    """'cuda' or 'cpu', decided by what actually works."""
    return verdict()["device"]


def use_fp16() -> bool:
    """
    Whether to run the encoder in half precision.

    Kept separate from `available()` because it changes the numbers that go
    into an index, not just the speed: a pre-Volta card can run CUDA fine and
    still be the wrong place for fp16.
    """
    v = verdict()
    return bool(v["ok"] and v["fp16"])


def precision() -> str:
    """The label recorded alongside an index, so backends are never mixed."""
    return "fp16" if use_fp16() else "fp32"


def describe() -> str:
    """One line for the UI."""
    v = verdict()
    if v["ok"]:
        return f"Using {v['name']} ({v['capability']}, {precision()})."
    return f"Using the CPU — {v['reason']}."

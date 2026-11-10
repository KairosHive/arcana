# loras.py -- the texture LoRAs on this machine, for Alchemy's texture LoRA method
#
# A LoRA is a Stable Diffusion 1.5 checkpoint folder holding
# pytorch_lora_weights.safetensors, as latent-lab's lora.py and diffusers'
# trainers write it (sd15-lora-yellow/step-002000). It is named by its path
# under the folder it was found in, so a name reads the same in a recipe, a
# film and the tab.
#
# Where they are looked for, in order:
#
#   ARCANA_LORA_DIRS        any folders, separated like PATH
#   <data dir>/loras        Arcana's own
#   ~/latent-lab-models     where latent-lab trains them
#
# The hub mirror inside latent-lab-models (hub/...) holds copies of the same
# weights and is skipped.

from __future__ import annotations

import os
import pathlib

WEIGHTS = "pytorch_lora_weights.safetensors"
TRIGGER = "in goofi style"   # the word latent-lab's lora.py ties a training set to by default


def roots() -> list[pathlib.Path]:
    from . import paths as _paths

    found = [pathlib.Path(p).expanduser() for p in os.environ.get("ARCANA_LORA_DIRS", "").split(os.pathsep) if p]
    found += [pathlib.Path(_paths.data_dir()) / "loras", pathlib.Path("~/latent-lab-models").expanduser()]
    return [r for r in found if r.is_dir()]


def available() -> dict[str, pathlib.Path]:
    """Every LoRA checkpoint, by name, its folder as the value. Later checkpoints of a run come first."""
    out: dict[str, pathlib.Path] = {}
    for root in roots():
        for weights in root.rglob(WEIGHTS):
            folder = weights.parent
            rel = folder.relative_to(root)
            if rel.parts and rel.parts[0] == "hub":
                continue
            out.setdefault(rel.as_posix(), folder)
    return dict(sorted(out.items(), key=lambda kv: (kv[0].rsplit("/", 1)[0], -_step(kv[0]))))


def _step(name: str) -> int:
    tail = name.rsplit("/", 1)[-1]
    digits = "".join(ch for ch in tail if ch.isdigit())
    return int(digits) if digits else 0


def options() -> tuple[str, ...]:
    return ("none",) + tuple(available())


def locate(name: str) -> pathlib.Path:
    folder = available().get(name)
    if folder is None:
        raise ValueError(f"No LoRA called {name!r}. LoRAs: {', '.join(available()) or 'none found'}. "
                         f"Put one under {os.path.join('<data dir>', 'loras')} or name its folder in ARCANA_LORA_DIRS.")
    return folder


def prompt(name: str) -> str | None:
    """The caption a LoRA trained in Arcana draws its set with (arcana-lora.json), or None."""
    import json

    try:
        with open(locate(name) / "arcana-lora.json", encoding="utf-8") as f:
            return json.load(f).get("prompt") or None
    except (ValueError, OSError):
        return None


def trigger(name: str) -> str:
    """The words a LoRA was trained to answer: from the run's arcana-lora.json or latent-lab.json,
    TRIGGER when neither says."""
    import json

    try:
        folder = locate(name)
    except ValueError:
        return TRIGGER
    for d in (folder, folder.parent):
        for meta in ("arcana-lora.json", "latent-lab.json"):
            try:
                with open(d / meta, encoding="utf-8") as f:
                    t = json.load(f).get("trigger")
                if t:
                    return t
            except (OSError, ValueError):
                continue
    return TRIGGER

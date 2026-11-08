# system.py -- the ops by which Arcana describes itself
#
# orient is the first page anyone reads, person or model. op list and help are
# generated from the registry, so they cannot fall behind the ops they list.

from __future__ import annotations

import sys
import textwrap

from pydantic import BaseModel, Field

from . import archive, grammar, registry
from .registry import Args, Context, Result, TextResult, op, positional

try:
    from .. import __version__
    from .. import core
    from .. import paths as _paths
except ImportError:                                   # pragma: no cover
    from arcana import __version__, core
    from arcana import paths as _paths


ABOUT = """\
Arcana searches the person's own picture archive by what the pictures show, on
this machine, against indexes the app built. Nothing here edits or moves an
original. Pictures are named by handle, dataset:key (japan:1423).

Learn it in three steps: orient, then `op list`, then `<op> --help`.
"""

SEARCHING = """\
How to search. CLIP, which ranks the pictures, sees concrete visual things:
objects, places, light, weather, colour, texture, composition, photographic
technique. It misses abstractions, so translate them ('melancholy' becomes
'empty bench in rain, overcast, muted colours'). Give each idea several
phrasings, and use negatives for things that keep intruding. Survey a dataset
before planning a search, so the plan fits what the archive holds.

Every search facet comes back marked strong, fair or weak. Weak means the
archive has little for that idea. Say so plainly rather than presenting the
results as a match.
"""


def _gpu_line() -> str:
    """The card, without importing torch: nvidia-smi answers in a fraction of a second."""
    try:
        from .. import gpu as _gpu
    except ImportError:                               # pragma: no cover
        import gpu as _gpu
    if "torch" in sys.modules:                        # already paid for, so ask torch
        return "GPU: " + _gpu.describe()
    hw = _gpu.hardware()
    if not hw.get("nvidia"):
        return "GPU: none found, so models run on the CPU."
    vram = f", {hw['vram_mb'] // 1024} GB" if hw.get("vram_mb") else ""
    return f"GPU: {hw['name']}{vram}."


def _ops_by_noun() -> str:
    nouns: dict = {}
    singles = []
    for o in registry.all_ops():
        w = o.name.split()
        if len(w) == 1 or w[0] in ("op", "settings"):
            singles.append(o.name)
        else:
            nouns.setdefault(w[0], []).append(" ".join(w[1:]))
    parts = [f"{n} ({', '.join(v)})" for n, v in nouns.items()]
    return "Ops by noun: " + ", ".join(parts) + ". Also " + ", ".join(singles) + "."


class OrientArgs(Args):
    pass


@op("orient", "read", OrientArgs, TextResult,
    doc="""
    What Arcana is and what is on this machine. Read it first. It lists the
    datasets by id, says whether pictures can reach you, names the graphics
    card, lists the ops by noun, and says how to write a search.
    """,
    examples=("orient",))
def _orient(a: OrientArgs, ctx: Context) -> TextResult:
    found = core.find_datasets()
    if found:
        names = [archive.dataset_id(d["name"], d["modality"])
                 + ("" if d["layouts"] else " (no map)") for d in found]
        ds_line = ("Datasets: " + ", ".join(names)
                   + ". `dataset list` gives their sizes and encoders.")
    else:
        ds_line = "Datasets: none yet. The person builds one from the Datasets tab."
    if ctx.actor == "agent":
        mode = archive.mode_line()
    else:
        mode = ("Pictures: you see them as files on this machine. Add --sheets before the "
                "op to draw contact sheets for searches.")
    try:
        from .. import boards as _boards
        n_boards = len(_boards.listing())
    except Exception:
        n_boards = 0
    body = "\n".join([
        ABOUT,
        ds_line,
        mode,
        _gpu_line(),
        f"Boards: {n_boards}.",
        "",
        *textwrap.wrap(_ops_by_noun(), grammar.WIDTH),
        "",
        SEARCHING,
    ])
    return TextResult(body=body.rstrip() + "\n")


# ─────────────────────────── op list ───────────────────────────

class OpRow(BaseModel):
    name: str
    kind: str
    summary: str
    missing: list[str] = Field(default_factory=list)   # flags this Arcana lacks
    allowed: bool = True                               # may this caller run it


class OpListResult(Result):
    ops: list[OpRow]

    def text(self) -> str:
        width = max((len(r.name) for r in self.ops), default=10) + 2
        lines = []
        for r in self.ops:
            tail = ""
            if r.missing:
                tail = "  (needs " + ", ".join(r.missing) + ")"
            elif not r.allowed:
                tail = "  (for the person)"
            lines.append(f"{r.name:<{width}}{r.kind:<8}{r.summary}{tail}")
        return "\n".join(lines) + "\n\n`<op> --help` for any of them.\n"


class OpListArgs(Args):
    noun: str | None = positional(0, None, description="Only the ops of this noun, e.g. search.")


@op("op list", "read", OpListArgs, OpListResult,
    doc="""
    Every op, one line each, with its kind and what it does. An op this Arcana
    cannot run is marked with what it needs, so you learn what is missing
    before you try.
    """,
    examples=("op list", "op list search"))
def _op_list(a: OpListArgs, ctx: Context) -> OpListResult:
    rows = []
    for o in registry.all_ops():
        if a.noun and o.name.split()[0] != a.noun:
            continue
        rows.append(OpRow(name=o.name, kind=o.kind, summary=o.summary,
                          missing=[f for f in sorted(o.needs) if not ctx.has(f)],
                          allowed=ctx.actor in o.actors))
    if a.noun and not rows:
        nouns = sorted({o.name.split()[0] for o in registry.all_ops()})
        raise ValueError(f"No ops for {a.noun!r}. Nouns: {', '.join(nouns)}.")
    return OpListResult(ops=rows)


# ─────────────────────────── help ───────────────────────────

class HelpArgs(Args):
    op: list[str] = positional(0, [], max_length=3,
                               description="The op's name, e.g. search text. Empty for the grammar.")


@op("help", "read", HelpArgs, TextResult,
    doc="""
    An op's help page, the same one `<op> --help` prints. With no op, how a
    line is written.
    """,
    examples=("help", "help search text"))
def _help(a: HelpArgs, ctx: Context) -> TextResult:
    if not a.op:
        return TextResult(body=grammar.GRAMMAR)
    found, rest = grammar.resolve(a.op)
    if rest:
        raise ValueError(f"`{' '.join(a.op)}` is not an op. Did you mean `help {found.name}`?")
    return TextResult(body=grammar.help_text(found))


# ─────────────────────────── settings ───────────────────────────

class SettingsResult(Result):
    version: str
    data_dir: str
    agent_config: str
    pixels: bool
    flags: dict[str, bool]

    def text(self) -> str:
        on = [f for f, v in self.flags.items() if v]
        off = [f for f, v in self.flags.items() if not v]
        return "\n".join([
            f"Arcana {self.version}",
            f"Data: {self.data_dir}",
            f"Agent settings: {self.agent_config}",
            "Pixels: " + ("on, so agents receive thumbnails" if self.pixels
                          else "off, so agents work blind"),
            "Capabilities on: " + (", ".join(on) or "none"),
            "Capabilities off: " + (", ".join(off) or "none"),
        ]) + "\n"


class SettingsShowArgs(Args):
    pass


@op("settings show", "read", SettingsShowArgs, SettingsResult,
    doc="""
    Where Arcana keeps its data, and what this Arcana can do. It shows whether
    the person lets agents see thumbnails and which capabilities are present
    (gpu, diffusers, modflows, ui, frozen). Checking the GPU loads PyTorch, so
    the first call takes a few seconds.
    """,
    examples=("settings show",))
def _settings_show(a: SettingsShowArgs, ctx: Context) -> SettingsResult:
    cfg = archive.settings()
    flags = {f: ctx.has(f) for f in registry.FLAG_PROBES if f != "pixels"}
    return SettingsResult(version=__version__, data_dir=_paths.data_dir(),
                          agent_config=cfg["path"], pixels=cfg["pixels"], flags=flags)

# style.py -- restyling and generating pictures
#
# Texture transfer keeps every pixel of the target and borrows how much detail
# the source carries at each scale. The other three regenerate the target with
# sd-turbo or Stable Diffusion 1.5: img2img aimed by words CLIP finds for the
# source, IP-Adapter conditioned on the source image itself, and IP-Adapter
# with a ControlNet holding the result to a map of the target. `generate`
# makes a picture from words, or redraws one toward words.
#
# Every result is a new file in output/, recorded in the ledger (derived for a
# restyle, generated for a generation), and badged as such on any board.

from __future__ import annotations

import os
import time
from typing import Literal

from PIL import Image
from pydantic import BaseModel, Field

from . import archive
from .colour import item_arg, preview
from .effects import run_effect
from .registry import Args, Attachment, Context, Result, op, positional

try:
    from .. import ledger as _ledger
    from .. import paths as _paths
except ImportError:                                   # pragma: no cover
    import ledger as _ledger
    import paths as _paths

StyleMethod = Literal["texture", "img2img", "ip_adapter", "controlnet"]
ControlKind = Literal["depth", "canny", "luminance"]


class StyleInfo(BaseModel):
    id: str
    label: str
    needs: list[str]
    blurb: str


STYLE_METHODS = (
    StyleInfo(id="texture", label="Texture, keeps your pixels", needs=[],
              blurb="Keeps every pixel of the target and borrows how much detail the source "
                    "carries at each scale. Nothing is invented and the composition is exact. "
                    "Subtle between two photographs, strong from a painting or a film scan."),
    StyleInfo(id="ip_adapter", label="IP-Adapter, strongest", needs=["diffusers"],
              blurb="Regenerates the target conditioned on the source image itself. It "
                    "carries what language cannot name. Around 0.4 to 0.6 keeps the subject "
                    "recognisable. Higher and the source takes over."),
    StyleInfo(id="img2img", label="img2img, from words", needs=["diffusers"],
              blurb="Regenerates the target, aimed by words CLIP finds for the source. It "
                    "never sees the source itself, so it carries mood rather than a look."),
    StyleInfo(id="controlnet", label="IP-Adapter + ControlNet, style along a map",
              needs=["diffusers"],
              blurb="IP-Adapter carries the source's look and a ControlNet holds the result to "
                    "a map of the target (depth, edges or lightness), so texture follows form "
                    "instead of lying flat. It keeps a face a face at strengths where "
                    "IP-Adapter alone dissolves it."),
)
BY_ID = {m.id: m for m in STYLE_METHODS}


def _st():
    try:
        from .. import style_transfer as _s
    except ImportError:                               # pragma: no cover
        import style_transfer as _s
    return _s


class MethodsResult(Result):
    methods: list[StyleInfo]
    missing: dict[str, list[str]] = Field(default_factory=dict)

    def text(self) -> str:
        lines = []
        for m in self.methods:
            gap = self.missing.get(m.id)
            lines.append(f"{m.id:<11} {m.label}" + (f"  (needs {', '.join(gap)})" if gap else ""))
            lines.append(f"            {m.blurb}")
        return "\n".join(lines) + "\n"


class NoArgs(Args):
    pass


@op("style methods", "read", NoArgs, MethodsResult,
    doc="The style transfer methods, with what each does and what it needs.",
    examples=("style methods",))
def _style_methods(a: NoArgs, ctx: Context) -> MethodsResult:
    missing = {m.id: [n for n in m.needs if not ctx.has(n)] for m in STYLE_METHODS}
    return MethodsResult(methods=list(STYLE_METHODS), missing={k: v for k, v in missing.items() if v})


# ─────────────────────────── style transfer ───────────────────────────

class StyleResult(Result):
    handle: str
    path: str
    file: str
    method: str
    label: str
    device: str
    elapsed: float
    width: int
    height: int
    target: str
    source: str
    map_path: str | None = None

    def text(self) -> str:
        out = (f"{self.handle}  {self.label}, {self.target} in the style of {self.source}.\n"
               f"  {self.width} x {self.height} · {self.device} · {self.elapsed:.1f} s · {self.file}\n")
        if self.map_path:
            out += f"  The map ControlNet followed: {self.map_path}\n"
        return out + "  Derived, so it is marked as such on any board. The original is untouched.\n"


class StyleTransferArgs(Args):
    target: str = item_arg(0, "The picture to restyle: a handle, or out: for a derived one.")
    source: str = Field(description="The picture whose look to take.",
                        json_schema_extra={"metavar": "ITEM"})
    method: StyleMethod = Field("texture", description="texture, img2img, ip_adapter or controlnet.")
    strength: float = Field(0.6, ge=0.0, le=1.0,
                            description="How loudly the source speaks. For texture, how much "
                                        "detail it borrows.")
    control: ControlKind = Field("depth", description="controlnet only: the map of the target "
                                                      "to hold the result to.")
    control_strength: float = Field(0.9, ge=0.0, le=2.0,
                                    description="controlnet only: how strictly the map is obeyed.")
    prompt: str | None = Field(None, description="Words to steer the diffusion methods.")
    steps: int | None = Field(None, ge=1, le=50, description="Diffusion steps. Each method has "
                                                             "its own default.")
    seed: int = Field(2222, description="The seed, for a result you can repeat.")
    background: bool = Field(False, description="Return a job id at once instead of waiting.")


@op("style transfer", "effect", StyleTransferArgs, StyleResult,
    modalities=("image",),
    doc="""
    Apply the look of one picture to another. `style methods` says what each
    method does. The diffusion methods download their weights on first use,
    which takes a few minutes, and run far faster on a graphics card. The
    result is a new file with an out: handle, marked as derived.
    """,
    examples=("style transfer japan:12 --source japan:7 --method texture --strength 0.5",))
def _style_transfer(a: StyleTransferArgs, ctx: Context) -> Result:
    for need in BY_ID[a.method].needs:
        if not ctx.has(need):
            raise ValueError(f"{a.method} needs the diffusers library, which this Arcana lacks. "
                             f"texture runs everywhere.")
    target = archive.resolve_item(a.target, ctx.actor)
    source = archive.resolve_item(a.source, ctx.actor)
    for it in (target, source):
        if not os.path.exists(it.path):
            raise ValueError(f"{it.ref}'s file is not where the index says ({it.path}).")

    def compute(progress):
        st = _st()
        started = time.time()
        src = Image.open(source.path).convert("RGB")
        tgt = Image.open(target.path).convert("RGB")
        cb = progress                     # style_transfer reports (fraction, message) too
        cmap = None
        kw = {"prompt": a.prompt} if a.prompt else {}
        if a.steps:
            kw["steps"] = a.steps
        if a.method == "texture":
            out = st.texture_transfer(src, tgt, strength=a.strength)
        elif a.method == "img2img":
            out = st.img2img_transfer(src, tgt, strength=a.strength, seed=a.seed, progress=cb, **kw)
        elif a.method == "ip_adapter":
            out = st.ip_adapter_transfer(src, tgt, scale=a.strength, seed=a.seed, progress=cb,
                                         **{"prompt": a.prompt or "", **({"steps": a.steps} if a.steps else {})})
        else:
            out, cmap = st.controlnet_transfer(src, tgt, control=a.control, scale=a.strength,
                                               control_scale=a.control_strength, seed=a.seed,
                                               progress=cb,
                                               **{"prompt": a.prompt or "",
                                                  **({"steps": a.steps} if a.steps else {})})
        out_dir = _paths.edited_dir("generative")
        name = _paths.fit_filename(out_dir, os.path.splitext(os.path.basename(target.path))[0],
                                   os.path.splitext(os.path.basename(source.path))[0],
                                   f"{a.method}_{time.strftime('%Y%m%d_%H%M%S')}", ".jpg")
        path = os.path.join(out_dir, name)
        out.save(path, quality=95)
        map_path = None
        if cmap is not None:
            map_path = os.path.splitext(path)[0] + f"_{a.control}-map.jpg"
            cmap.convert("RGB").save(map_path, quality=90)
        try:
            from .. import gpu as _g
        except ImportError:                           # pragma: no cover
            import gpu as _g
        # Texture is OpenCV arithmetic and never touches the GPU.
        device = "CPU" if a.method == "texture" else ("GPU" if _g.available() else "CPU")
        return {"path": path, "map": map_path, "size": out.size, "device": device,
                "elapsed": time.time() - started}

    def record(tx, p):
        label = {"texture": "Texture", "img2img": "img2img", "ip_adapter": "IP-Adapter",
                 "controlnet": f"ControlNet, {a.control}"}[a.method]
        entry = _ledger.record(tx, p["path"], op="style transfer",
                               args={"method": a.method, "strength": a.strength,
                                     "control": a.control if a.method == "controlnet" else None,
                                     "seed": a.seed, "prompt": a.prompt},
                               sources=[s for s in (target.handle, source.handle) if s],
                               derived_from=target.handle)
        res = StyleResult(handle=entry["handle"], path=p["path"], file=os.path.basename(p["path"]),
                          method=a.method, label=label, device=p["device"], elapsed=p["elapsed"],
                          width=p["size"][0], height=p["size"][1],
                          target=target.handle or a.target, source=source.handle or a.source,
                          map_path=p["map"])
        if ctx.sheets:
            res.attachments.append(preview(Image.open(p["path"]), caption=res.handle))
        return res

    return run_effect(ctx, f"style transfer {a.target}", compute, record, a.background)


# ─────────────────────────── control map ───────────────────────────

class MapResult(Result):
    handle: str
    kind: str
    width: int
    height: int
    withheld: bool = False

    def text(self) -> str:
        out = f"The {self.kind} map of {self.handle}, {self.width} x {self.height}."
        if self.withheld:
            out += " It stays on this machine: the person has not allowed pictures to reach agents."
        return out + "\n"


class ControlMapArgs(Args):
    handle: str = item_arg(0, "The picture to map: a handle, or out: for a derived one.")
    kind: ControlKind = Field("depth", description="depth, canny or luminance.")


@op("style control-map", "read", ControlMapArgs, MapResult,
    modalities=("image",),
    doc="""
    The map a ControlNet will read from a picture. Judge it before a
    transfer. A wrong depth map explains a wrong result. Depth loads a depth
    model the first time. Luminance needs no model.
    """,
    examples=("style control-map japan:12 --kind luminance",))
def _control_map(a: ControlMapArgs, ctx: Context) -> MapResult:
    it = archive.resolve_item(a.handle, ctx.actor)
    m = _st().control_map(a.kind, Image.open(it.path).convert("RGB"))
    res = MapResult(handle=it.handle or a.handle, kind=a.kind, width=m.size[0], height=m.size[1],
                    withheld=not ctx.pixels)
    if ctx.pixels:
        res.attachments.append(preview(m, caption=f"{a.kind} map"))
    return res


# ─────────────────────────── generate ───────────────────────────

class GenerateResult(Result):
    handle: str
    path: str
    file: str
    prompt: str
    init: str | None = None
    seed: int
    elapsed: float
    width: int
    height: int
    board: str | None = None

    def text(self) -> str:
        made = f"redrawn from {self.init}" if self.init else "from words alone"
        out = (f"{self.handle}  generated {made}: \"{self.prompt}\"\n"
               f"  {self.width} x {self.height} · seed {self.seed} · {self.elapsed:.1f} s · {self.file}\n"
               f"  Marked as generated wherever it goes.\n")
        if self.board:
            out += f"  Added to `{self.board}`.\n"
        return out


class GenerateArgs(Args):
    prompt: str = Field(description="What the picture shows, in concrete words.")
    init: str | None = Field(None, description="A picture to redraw toward the prompt, by handle.",
                             json_schema_extra={"metavar": "ITEM"})
    strength: float = Field(0.72, ge=0.05, le=1.0,
                            description="With --init, how far to move from it. 1 replaces it.")
    steps: int = Field(4, ge=1, le=8, description="sd-turbo steps. Four is its sweet spot.")
    seed: int = Field(2222, description="The seed, for a picture you can repeat.")
    width: int = Field(512, ge=256, le=1024, description="Without --init, the width in pixels.")
    height: int = Field(512, ge=256, le=1024, description="Without --init, the height in pixels.")
    for_board: str | None = Field(None, description="Add the result to this board as a "
                                                    "generated picture.",
                                  json_schema_extra={"complete": "board", "metavar": "BOARD"})
    section: str | None = Field(None, description="With --for-board, the section to place it in.")
    role: Literal["support", "texture", "detail", "counterpoint"] = Field(
        "support", description="With --for-board, its role. Never hero.")
    note: str | None = Field(None, description="With --for-board, why it is there.")
    folder: str = Field("generated", description="Where under output/ to write it.")
    background: bool = Field(False, description="Return a job id at once instead of waiting.")


MAX_GENERATED = 2


@op("generate", "effect", GenerateArgs, GenerateResult,
    modalities=("image",),
    doc="""
    Make a picture with sd-turbo, from words or by redrawing a picture toward
    words. Use it to fill an angle the archive is weak on, sparingly: a board
    takes at most two generated pictures, and never as its hero. The result is
    marked as generated wherever it goes. Loads the model the first time.
    """,
    examples=('generate --prompt "rain on a tin roof, close up, grey light" --steps 2',),
    needs=("diffusers",))
def _generate(a: GenerateArgs, ctx: Context) -> Result:
    from . import board as _board
    init = archive.resolve_item(a.init, ctx.actor) if a.init else None
    if a.for_board:
        rec = _board.load(a.for_board)
        _board.check_section(rec, a.section)
        made = sum(1 for d in (rec.get("details") or {}).values() if d.get("kind") == "generated")
        if made >= MAX_GENERATED:
            raise ValueError(f"`{a.for_board}` already has {made} generated pictures, and a board "
                             f"takes at most {MAX_GENERATED}. Find the rest in the archive.")
    out_root = _paths.subdir("output")
    out_dir = _paths.safe_join(out_root, *[p for p in a.folder.replace("\\", "/").split("/") if p])
    if out_dir is None:
        raise ValueError(f"{a.folder!r} is not a folder under output/.")

    def compute(progress):
        try:
            from .. import generate as _gen
        except ImportError:                           # pragma: no cover
            import generate as _gen
        started = time.time()
        if init is not None:
            img = _gen.img2img(Image.open(init.path), a.prompt, strength=a.strength,
                               steps=a.steps, seed=a.seed, progress=progress)
        else:
            img = _gen.txt2img(a.prompt, steps=a.steps, seed=a.seed, width=a.width,
                               height=a.height, progress=progress)
        _paths.ensure_dir(out_dir)
        stem = "".join(c if c.isalnum() else "-" for c in a.prompt.lower())[:48].strip("-") or "picture"
        path = os.path.join(out_dir, f"{stem}_{a.seed}_{time.strftime('%Y%m%d_%H%M%S')}.png")
        img.save(path)
        return {"path": path, "size": img.size, "elapsed": time.time() - started}

    def record(tx, p):
        entry = _ledger.record(tx, p["path"], op="generate", kind="generated",
                               args={"prompt": a.prompt, "strength": a.strength if init else None,
                                     "steps": a.steps, "seed": a.seed},
                               sources=[init.handle] if init and init.handle else [],
                               derived_from=init.handle if init else None)
        res = GenerateResult(handle=entry["handle"], path=p["path"], file=os.path.basename(p["path"]),
                             prompt=a.prompt, init=(init.handle or a.init) if init else None,
                             seed=a.seed, elapsed=p["elapsed"], width=p["size"][0],
                             height=p["size"][1], board=a.for_board)
        if a.for_board:
            rec = _board.load(a.for_board)
            item = archive.Item(ref=entry["handle"], path=p["path"], handle=entry["handle"],
                                kind="generated", derived_from=entry.get("derived_from"))
            _board.add_item(rec, item, section=a.section, role=a.role, note=a.note)
            _board.put(ctx, rec, tx=tx)
        if ctx.sheets:
            res.attachments.append(preview(Image.open(p["path"]), caption=res.handle))
        return res

    return run_effect(ctx, f"generate {a.prompt[:40]}", compute, record, a.background)

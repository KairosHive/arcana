# ops/pattern.py -- procedural editing as ops: render a pattern, list the presets
#
# A pattern places frames by rule: sources interlaced in time and in space,
# flashes on rhythms, images hidden in complementary pairs (arcana/pattern.py,
# VIDEO_ROADMAP.md 3.5). `pattern make` renders one to a video. Its sources are
# named the way every op names media, by handle, so an agent can write a
# pattern from what a search returned.

from __future__ import annotations

import datetime
import json
import os

from pydantic import Field, field_validator

from . import archive
from .effects import run_effect
from .registry import Args, Context, Result, op

try:
    from .. import pattern as _pattern
    from .. import paths as _paths
except ImportError:                                   # pragma: no cover
    import pattern as _pattern
    import paths as _paths


class PatternMade(Result):
    path: str
    frames: int
    fps: float
    seconds: float
    flashes_per_second: int = 0
    over_flash_limit: bool = False
    flash_at: float | None = None

    def text(self) -> str:
        t = f"Rendered {self.frames} frames, {self.seconds:.1f} s at {self.fps:g} fps.\n  {self.path}\n"
        if self.over_flash_limit:
            t += (f"  Flash meter: up to {self.flashes_per_second} flashes in one second (around "
                  f"{self.flash_at} s). More than three a second can trigger seizures in people with "
                  f"photosensitive epilepsy. Say so wherever this is shown.\n")
        else:
            t += f"  Flash meter: at most {self.flashes_per_second} flashes in any one second.\n"
        return t


class PatternMakeArgs(Args):
    spec: dict = Field(description="The pattern as a JSON object: fps, seconds, size, sources, voices, "
                                   "space, time. `pattern presets` shows examples.")
    name: str = Field("pattern", description="A name for the file.")
    background: bool = Field(False, description="Return a job id at once instead of waiting.")

    @field_validator("spec", mode="before")
    @classmethod
    def _spec(cls, v):
        if isinstance(v, str):
            try:
                v = json.loads(v)
            except ValueError:
                raise ValueError("--spec is a JSON object.") from None
        if not isinstance(v, dict):
            raise ValueError("--spec is a JSON object.")
        return v


def _resolved(spec: dict, actor: str) -> dict:
    """The spec with every source turned into a file path, a section reference or a colour."""
    out = dict(spec)
    sources = {}
    for name, ref in (spec.get("sources") or {}).items():
        ref = str(ref)
        if ref.startswith("#") and len(ref) == 7:
            sources[name] = ref
            continue
        sources[name] = archive.resolve_item(ref, actor).path
    out["sources"] = sources
    return out


@op("pattern make", "effect", PatternMakeArgs, PatternMade,
    modalities=("image", "video"),
    doc="""
    Render a pattern to a video in output/edited/video/<date>/. A pattern
    places frames by rule. Sources (clips, sections, pictures, flat colours)
    are interlaced frame by frame or line by line, flashed on rhythms that can
    run against each other, or hidden in pairs of frames that cancel at speed.
    The same pattern and seed always render the same video. The result carries
    a flash meter reading. Sources are handles (clips@video:12, japan:7) or
    colours (#000000).
    """,
    examples=('pattern make --spec \'{"seconds": 1, "size": [320, 240], "sources": {"A": "japan:1", '
              '"B": "japan:2"}, "voices": [{"pulse": {"every": 1}, "cut": ["A", "B"]}]}\'',))
def _pattern_make(a: PatternMakeArgs, ctx: Context) -> Result:
    spec = _resolved(a.spec, ctx.actor)
    _pattern.plan(spec)                         # refuse a bad pattern before any job starts
    slug = "".join(c if c.isalnum() or c in "-_" else "-" for c in a.name.strip().lower())[:40] or "pattern"
    path = os.path.join(_paths.edited_dir("video"),
                        f"{slug}__{datetime.datetime.now().strftime('%H%M%S')}.mp4")

    def compute(progress):
        made = _pattern.write(spec, path, progress)
        with open(os.path.splitext(path)[0] + ".json", "w", encoding="utf-8") as fh:
            json.dump({"pattern": a.spec, "made": {k: v for k, v in made.items() if k != "path"}}, fh, indent=1)
        return made

    def record(tx, made):
        flash = made["flash"]
        return PatternMade(path=made["path"], frames=made["frames"], fps=made["fps"],
                           seconds=made["frames"] / made["fps"], flashes_per_second=flash["max_per_second"],
                           over_flash_limit=flash["over"], flash_at=flash["at"])

    return run_effect(ctx, f"pattern make {a.name}", compute, record, a.background, kind="film")


class PresetsResult(Result):
    presets: dict

    def text(self) -> str:
        lines = ["Patterns to start from. Each is the part of a spec that makes the pattern; add fps, "
                 "seconds, size and sources A and B.", ""]
        for name, part in self.presets.items():
            lines.append(f"{name}\n  {json.dumps(part)}")
        return "\n".join(lines) + "\n"


class NoArgs(Args):
    pass


@op("pattern presets", "read", NoArgs, PresetsResult,
    doc="The patterns to start from: interlacing, polyrhythm, drifting lines, hidden images.",
    examples=("pattern presets",))
def _pattern_presets(a: NoArgs, ctx: Context) -> PresetsResult:
    return PresetsResult(presets=_pattern.PRESETS)

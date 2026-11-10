# ops/sequence.py -- cut clips together, as an op
#
# `sequence make` renders sections of video (and still pictures) one after
# another with their sound, and writes a CMX 3600 edit list beside the render,
# so the cut reopens in an editor on the original files. See arcana/sequence.py.

from __future__ import annotations

import datetime
import json
import os

from pydantic import Field

from . import archive
from .effects import run_effect
from .registry import Args, Context, Result, op

try:
    from .. import paths as _paths
    from .. import sequence as _sequence
except ImportError:                                   # pragma: no cover
    import paths as _paths
    import sequence as _sequence


class SequenceMade(Result):
    path: str
    edl: str
    clips: int
    seconds: float
    size: list[int]
    fps: float
    sound: bool

    def text(self) -> str:
        return (f"Cut {self.clips} clips into {self.seconds:.1f} s at {self.size[0]} x {self.size[1]}, "
                f"{'with' if self.sound else 'without'} sound.\n  {self.path}\n"
                f"  Edit list for Resolve or Premiere, on the original files: {self.edl}\n")


class SequenceMakeArgs(Args):
    clip: list[str] = Field(min_length=1, max_length=200,
                            description="A clip, repeated in order: a section or a whole video as "
                                        "dataset:key, or a picture, which is held as a still.",
                            json_schema_extra={"metavar": "ITEM"})
    name: str = Field("sequence", description="A name for the files.")
    notes: list[str] = Field(default_factory=list, description="A note for each clip, in the clips' order. "
                                                               "They go in the edit list.")
    still: float = Field(2.0, ge=0.2, le=30, description="Seconds a still picture is held.")
    sound: bool = Field(True, description="Keep the clips' sound. --no-sound renders silent.")
    width: int = Field(1280, ge=64, le=3840, description="Width in pixels.")
    height: int = Field(720, ge=64, le=2160, description="Height in pixels.")
    fps: int = Field(30, ge=6, le=60, description="Frames a second.")
    background: bool = Field(False, description="Return a job id at once instead of waiting.")


@op("sequence make", "effect", SequenceMakeArgs, SequenceMade,
    modalities=("image", "video"),
    doc="""
    Cut clips together in the order given, with their sound. The video goes
    to output/edited/video/<date>/. Each clip is a section or a whole video, by
    handle, or a picture held as a still. Sources of any shape are scaled to
    fill the frame. An edit list (CMX 3600) is written beside the video. It
    names the original files and their timecodes, so the cut opens in Resolve
    or Premiere at full quality.
    """,
    examples=("sequence make --clip japan:1 --clip japan:2 --still 0.5 --width 320 --height 240",))
def _sequence_make(a: SequenceMakeArgs, ctx: Context) -> Result:
    clips = []
    for n, ref in enumerate(a.clip):
        item = archive.resolve_item(ref, ctx.actor)
        clip = {"ref": item.path, "seconds": a.still}
        if n < len(a.notes) and a.notes[n]:
            clip["note"] = a.notes[n]
        clips.append(clip)
    spec = {"name": a.name, "fps": a.fps, "size": [a.width, a.height], "sound": a.sound, "clips": clips}
    _sequence.resolve(spec)                     # refuse a bad sequence before any job starts
    slug = "".join(c if c.isalnum() or c in "-_" else "-" for c in a.name.strip().lower())[:40] or "sequence"
    stem = os.path.join(_paths.edited_dir("video"), f"{slug}__{datetime.datetime.now().strftime('%H%M%S')}")

    def compute(progress):
        made = _sequence.render(spec, stem + ".mp4", progress)
        with open(stem + ".edl", "w", encoding="utf-8", newline="\r\n") as fh:
            fh.write(_sequence.edl(spec))
        with open(stem + ".json", "w", encoding="utf-8") as fh:
            json.dump({"sequence": spec, "made": made}, fh, indent=1)
        return made

    def record(tx, made):
        return SequenceMade(path=made["path"], edl=stem + ".edl", clips=made["clips"], seconds=made["seconds"],
                            size=made["size"], fps=made["fps"], sound=made["sound"])

    return run_effect(ctx, f"sequence make {a.name}", compute, record, a.background, kind="film")

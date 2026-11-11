# ops/film.py -- the open film, as state every caller shares
#
# The Video subtab edits one film at a time. That film lives in the data
# folder (state/film.json), and every edit to it is an op, so an agent's edit
# shows in the open page within a second (the version bumps, the page's event
# stream fires, the tab reloads it) and undo takes an edit back like any other
# write. `alchemy film save` keeps a copy under a name, and `film open` brings
# a kept one back. See VIDEO_ROADMAP.md, 3.7.

from __future__ import annotations

import json
import os
from typing import Literal

from pydantic import Field, field_validator

from . import archive
from .registry import Args, Context, Result, op, positional

try:
    from .. import paths as _paths
    from .. import ui_film as _ui
except ImportError:                                   # pragma: no cover
    import paths as _paths
    import ui_film as _ui

EASE = Literal["linear", "smooth", "in", "out", "hold"]


def _film():
    try:
        from ..alchemy import film
    except ImportError:                               # pragma: no cover
        from alchemy import film
    return film


def open_path() -> str:
    return os.path.join(_paths.ensure_dir(_paths.subdir("state")), "film.json")


def current() -> dict:
    """The open film's spec as last written, or an empty film."""
    try:
        with open(open_path(), encoding="utf-8") as fh:
            got = json.load(fh)
        return got if isinstance(got, dict) else _ui.full({})
    except (FileNotFoundError, ValueError):
        return _ui.full({})


def _write(ctx: Context, spec: dict) -> dict:
    ctx.tx.touch(open_path())
    path = open_path()
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(spec, fh, indent=1, ensure_ascii=False)
    os.replace(tmp, path)
    return spec


def _check_refs(spec: dict, ctx: Context) -> None:
    """Every picture, clip, material and sound the film names, resolved by the caller's rules: an
    agent names them by handle."""
    film = _film()
    for p in spec.get("pictures") or []:
        ref = str(p.get("ref", ""))
        if ref and not film.is_film(ref):
            archive.resolve_item(ref, ctx.actor)
    for k in spec.get("looks") or []:
        for m in (k.get("settings") or {}).get("materials") or []:
            if m:
                archive.resolve_item(str(m), ctx.actor)
    for a in spec.get("audio") or []:
        if a.get("ref"):
            archive.resolve_item(str(a["ref"]), ctx.actor)


class OpenFilm(Result):
    spec: dict
    length: float | None = None
    status: str = ""

    def text(self) -> str:
        s = self.spec
        lines = [f"The open film, {s.get('name', 'untitled film')}"
                 + (f", {self.length:.2f} s" if self.length else "") + "."]
        pics = sorted(s.get("pictures") or [], key=lambda p: float(p["t"]))
        for n, p in enumerate(pics, 1):
            extra = ""
            if p.get("out") is not None:
                extra = f"  plays {float(p.get('in') or 0):.2f} to {float(p['out']):.2f} s at {float(p.get('speed') or 1):g}x"
            lines.append(f"  key {n}  {float(p['t']):7.2f} s  {p['ref']}{extra}")
        for k in s.get("looks") or []:
            lines.append(f"  look   {float(k['t']):7.2f} s  {(k.get('settings') or {}).get('method') or 'morph only'}")
        for path, keys in (s.get("lanes") or {}).items():
            lines.append(f"  lane   {path}: " + ", ".join(f"{float(k['v']):g} at {float(k['t']):.2f} s" for k in keys))
        for a in s.get("audio") or []:
            lines.append(f"  sound  {float(a.get('t') or 0):7.2f} s  {a['ref']}")
        if self.status:
            lines.append("  " + self.status)
        return "\n".join(lines) + "\n"


def _result(spec: dict) -> OpenFilm:
    f, why = _ui.normal(spec)
    return OpenFilm(spec=spec, length=f["length"] if f else None,
                    status="" if f or len(spec.get("pictures") or []) < 2 else why)


# ─────────────────────────── reading ───────────────────────────

class FilmCurrentArgs(Args):
    pass


@op("film current", "read", FilmCurrentArgs, OpenFilm,
    doc="""
    The film open in the Video subtab. It lists its keys in time order
    (pictures, clips and kept films, numbered as `film remove` takes them), its
    looks, its lanes and its sounds. The spec it returns is what `film set`
    takes.
    """,
    examples=("film current",))
def _film_current(a: FilmCurrentArgs, ctx: Context) -> OpenFilm:
    return _result(current())


# ─────────────────────────── replacing it ───────────────────────────

class FilmOpenArgs(Args):
    film: str | None = positional(0, None, description="A kept film, by name. Without one, a new empty film.",
                                  json_schema_extra={"metavar": "FILM"})
    name: str | None = Field(None, description="The new film's name, when no kept film is named.")


@op("film open", "write", FilmOpenArgs, OpenFilm,
    doc="""
    Open a kept film in the Video subtab, or a new empty one, in place of the
    film open now. Undo brings the previous one back.
    """,
    examples=("film open --name \"night walk\"",))
def _film_open(a: FilmOpenArgs, ctx: Context) -> OpenFilm:
    spec = _film().load(a.film) if a.film else _ui.full({"name": (a.name or "untitled film").strip()})
    return _result(_write(ctx, spec))


class FilmSetArgs(Args):
    spec: dict = Field(description="The whole film as a JSON object, as `film current` returns it.")

    @field_validator("spec", mode="before")
    @classmethod
    def _spec(cls, v):
        if isinstance(v, str):
            try:
                v = json.loads(v)
            except ValueError as e:
                raise ValueError(f"--spec is a JSON object: {e}")
        if not isinstance(v, dict):
            raise ValueError("--spec is a JSON object.")
        return v


@op("film set", "write", FilmSetArgs, OpenFilm,
    modalities=("image", "audio", "video"),
    doc="""
    Replace the open film with a whole spec. The Video subtab sends each edit
    this way. The film may be unfinished (fewer than two keys); it is checked
    in full when it renders.
    """,
    examples=("film set --spec '{\"name\": \"walk\", \"pictures\": [{\"t\": 0, \"ref\": \"japan:12\"}]}'",))
def _film_set(a: FilmSetArgs, ctx: Context) -> OpenFilm:
    spec = _ui.full(a.spec)
    _check_refs(spec, ctx)
    return _result(_write(ctx, spec))


# ─────────────────────────── edits ───────────────────────────

class FilmAddArgs(Args):
    item: str = positional(0, description="A picture, a clip (a section or a whole video) or a kept film "
                                          "(film:<name>), by handle.", json_schema_extra={"metavar": "ITEM"})
    at: float = Field(0.0, ge=0, description="When, in seconds. A time taken or inside a playing key goes "
                                              "after everything.")
    start: float | None = Field(None, ge=0, description="A clip or film: where it starts playing, in seconds "
                                                      "of its own (its in point).")
    end: float | None = Field(None, ge=0, description="A clip or film: where it stops (its out point).")
    speed: float | None = Field(None, ge=0.1, le=8, description="A clip or film: how fast it plays.")
    ease: EASE = Field("linear", description="The curve of the passage that leaves this key.")


@op("film add", "write", FilmAddArgs, OpenFilm,
    modalities=("image", "video"),
    doc="""
    Put a picture, a clip or a kept film on the open film's picture lane. A
    clip plays between --start and --end with its sound, and the film morphs into
    it and out of it.
    """,
    examples=("film add japan:12 --at 0", "film add japan:7 --at 4 --ease smooth"))
def _film_add(a: FilmAddArgs, ctx: Context) -> OpenFilm:
    film = _film()
    ref = a.item.strip()
    if not film.is_film(ref):
        archive.resolve_item(ref, ctx.actor)
    spec, t = _ui.add_picture(current(), ref, a.at)
    pics = sorted(spec["pictures"], key=lambda p: float(p["t"]))
    key = next(p for p in pics if p["ref"] == ref and abs(float(p["t"]) - t) < 1e-6)
    key["ease"] = a.ease
    if film.plays(ref):
        for name, v in (("in", a.start), ("out", a.end), ("speed", a.speed)):
            if v is not None:
                key[name] = v
        film.held_key(key, 0)
    spec["pictures"] = pics
    return _result(_write(ctx, spec))


class FilmRemoveArgs(Args):
    key: int = positional(0, ge=1, description="The key's number, in time order, as `film current` lists it.")


@op("film remove", "write", FilmRemoveArgs, OpenFilm,
    doc="Take one key off the open film's picture lane.",
    examples=("film remove 2",))
def _film_remove(a: FilmRemoveArgs, ctx: Context) -> OpenFilm:
    spec = current()
    pics = sorted(spec.get("pictures") or [], key=lambda p: float(p["t"]))
    if a.key > len(pics):
        raise ValueError(f"The open film has {len(pics)} key{'s' if len(pics) != 1 else ''}. `film current` lists them.")
    pics.pop(a.key - 1)
    spec["pictures"] = pics
    spec.pop("length", None)
    return _result(_write(ctx, spec))


class FilmLaneArgs(Args):
    path: str = positional(0, description="The lane: knobs.<key>, finish.<key>, weights.<n>, zoom, amount, "
                                          "mask.<refinement> or mask.<n>.<number>.",
                           json_schema_extra={"metavar": "LANE"})
    at: float = Field(ge=0, description="When, in seconds.")
    value: float = Field(description="The lane's number at that time.")
    ease: EASE = Field("linear", description="The curve toward the next key.")


@op("film lane", "write", FilmLaneArgs, OpenFilm,
    doc="""
    Key one number of the open film at a time. The number is a knob, a finish
    setting, a material's weight, the zoom, the method's amount or a mask's
    number. A key at the same time is replaced.
    """,
    examples=("film lane finish.saturation --at 6 --value -100 --ease smooth",))
def _film_lane(a: FilmLaneArgs, ctx: Context) -> OpenFilm:
    _film().check_path(a.path)
    spec = current()
    lanes = dict(spec.get("lanes") or {})
    keys = [k for k in lanes.get(a.path, []) if abs(float(k["t"]) - a.at) > 1e-6]
    keys.append({"t": round(a.at, 3), "v": a.value, "ease": a.ease})
    lanes[a.path] = sorted(keys, key=lambda k: float(k["t"]))
    spec["lanes"] = lanes
    return _result(_write(ctx, spec))


def _step_settings():
    from .alchemy import StepSettings
    return StepSettings


class FilmLookArgs(_step_settings()):
    at: float = Field(0.0, ge=0, description="When the look is reached, in seconds. A look at the same time "
                                              "is replaced.")
    material: list[str] = Field(default_factory=list, max_length=3, description="A material [1], [2], [3], "
                                "by handle, repeated.", json_schema_extra={"metavar": "MATERIAL"})
    weight: list[float] = Field(default_factory=list, max_length=3, description="Each material's weight.")
    ease: EASE = Field("smooth", description="The curve toward the next look.")


@op("film look", "write", FilmLookArgs, OpenFilm,
    modalities=("image",),
    doc="""
    Put an Alchemy step on the open film's look lane. It takes a method, its
    knobs, materials, a mask and a finish, as `alchemy preview` does. Between
    two looks every number moves from one to the other. Diffusion methods stay
    off clips: while a clip plays, a look with one shows the finish alone.
    """,
    examples=("film look --at 0 --method pixels.grain --material japan:7",
              "film look --at 8 --finish '{\"saturation\": -60}'"))
def _film_look(a: FilmLookArgs, ctx: Context) -> OpenFilm:
    from .alchemy import settings_of

    for m in a.material:
        archive.resolve_item(m, ctx.actor)
    look = settings_of(a)
    look["materials"] = list(a.material)
    if a.weight:
        look["weights"] = list(a.weight)
    spec = _ui.with_look(current(), a.at, look)
    spec["looks"] = [{**k, "ease": a.ease} if abs(float(k["t"]) - a.at) < 1e-6 else k for k in spec["looks"]]
    return _result(_write(ctx, spec))


class FilmAudioAddArgs(Args):
    item: str = positional(0, description="A sound, or a video whose track to use, by handle.",
                           json_schema_extra={"metavar": "ITEM"})
    at: float = Field(0.0, ge=0, description="When it starts in the film, in seconds.")
    start: float | None = Field(None, ge=0, description="Where in the sound it starts, in seconds.")
    end: float | None = Field(None, ge=0, description="Where in the sound it stops.")
    gain: float = Field(0.0, ge=-60, le=24, description="Loudness, in decibels.")
    fade_in: float = Field(0.0, ge=0, le=30, description="Seconds of fade at the start.")
    fade_out: float = Field(0.0, ge=0, le=30, description="Seconds of fade at the end.")


@op("film audio add", "write", FilmAudioAddArgs, OpenFilm,
    modalities=("audio", "video"),
    doc="""
    Lay a sound under the open film. It is a sound from an audio dataset or
    the track of a video, from --at, trimmed by --start and --end, with gain
    and fades. Clips on the picture lane are heard as they play as well.
    """,
    examples=("film audio add field@audio:212 --at 0 --gain -6 --fade-in 0.5 --fade-out 2",))
def _film_audio_add(a: FilmAudioAddArgs, ctx: Context) -> OpenFilm:
    archive.resolve_item(a.item, ctx.actor)
    spec = current()
    sound = {"t": round(a.at, 3), "ref": a.item.strip(), "gain": a.gain, "fade": [a.fade_in, a.fade_out]}
    if a.start is not None:
        sound["in"] = a.start
    if a.end is not None:
        sound["out"] = a.end
    spec["audio"] = sorted(list(spec.get("audio") or []) + [sound], key=lambda x: float(x.get("t") or 0))
    return _result(_write(ctx, spec))


class FilmKeepArgs(Args):
    name: str | None = Field(None, description="Keep it under this name. Defaults to the film's own.")


@op("film keep", "write", FilmKeepArgs, OpenFilm,
    modalities=("image", "audio", "video"),
    doc="""
    Keep the open film under its name, as `alchemy film save` does. A kept
    film renders by name, opens again, and plays inside another film.
    """,
    examples=("film keep --name \"night walk\"",))
def _film_keep(a: FilmKeepArgs, ctx: Context) -> OpenFilm:
    film = _film()
    spec = current()
    if a.name:
        spec = {**spec, "name": a.name.strip()}
        _write(ctx, spec)
    f = film.normalised(spec)
    ctx.tx.touch(film.film_path(f["name"]))
    film.save(f)
    return _result(spec)

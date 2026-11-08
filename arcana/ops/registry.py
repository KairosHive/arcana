# registry.py -- every action Arcana can perform, declared once
#
# An op is a name, a kind, a Pydantic model for its arguments, a model for its
# result, a function, a paragraph of documentation and some examples. From that
# one declaration come the command-line flags, `--help`, completion, the JSON
# Schema and the op list (grammar.py), and the one entry point every surface
# uses, `call()` below. See OPS_PLAN.md.
#
# Parsing the args model is the only validation. The CLI, the MCP server and
# the tests all produce a dict and hand it to the model, so a constraint written
# in a Field reaches the help text, the schema and the refusal from there.

from __future__ import annotations

import difflib
import importlib.util
import re
import sys
import textwrap
from dataclasses import dataclass, field
from typing import Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, SerializeAsAny, ValidationError

ACTORS = ("person", "cli", "agent")
KINDS = ("read", "write", "effect")

# Op names are noun then verb, lowercase words: "search text", "board section add".
NAME_RE = re.compile(r"^[a-z]+( [a-z][a-z-]*){0,2}$")


# ─────────────────────────── args and results ───────────────────────────

class Args(BaseModel):
    """Base for every op's arguments. Unknown fields are refused, never ignored."""
    model_config = ConfigDict(extra="forbid")


def positional(index: int, default=..., **kwargs):
    """
    A Field that is also accepted bare on the command line, in position `index`.

    A list-typed positional takes every remaining bare word, so it has to be the
    last one. Everything else about the field (bounds, description) is ordinary
    Pydantic, and reaches the help text and the schema like any other field.
    """
    extra = dict(kwargs.pop("json_schema_extra", None) or {})
    extra["positional"] = index
    return Field(default, json_schema_extra=extra, **kwargs)


class Attachment(BaseModel):
    """
    A picture a result carries: a contact sheet, a close look, a rendered board.

    `data` holds the bytes and stays out of JSON. Each surface decides what to do
    with them. The CLI writes them to a file and fills in `path`, MCP sends them
    as image content when the person allows pixels, and a blind agent gets a line
    saying a picture was withheld.
    """
    caption: str = ""
    mime: str = "image/jpeg"
    data: bytes = Field(default=b"", exclude=True, repr=False)
    path: str | None = None


class Result(BaseModel):
    """Base for every op's result. Subclasses declare their fields and render text."""
    attachments: list[Attachment] = Field(default_factory=list)

    def text(self) -> str:                           # pragma: no cover - abstract
        raise NotImplementedError


class TextResult(Result):
    """A result that is a page of prose: orient, help."""
    body: str

    def text(self) -> str:
        return self.body


# ─────────────────────────── refusals ───────────────────────────

class Refusal(Exception):
    """
    An op said no, in words the caller can act on.

    `reason` is one or two sentences in Arcana's terms. `help` is the op's
    generated help, so a caller who got the shape wrong sees the right shape
    beside the complaint. A refusal is an answer, never a crash: the CLI prints
    it and exits 1, MCP returns it as error content the model reads.
    """

    def __init__(self, op: str | None, reason: str, help: str = ""):
        super().__init__(reason)
        self.op = op
        self.reason = reason
        self.help = help

    def __str__(self) -> str:
        return self.reason

    def as_dict(self) -> dict:
        return {"op": self.op, "reason": self.reason, "help": self.help}


# ─────────────────────────── capability flags ───────────────────────────
# Modes are flags. An op declares what it needs, `op list` marks an op whose
# need is unmet, and call() refuses it with the reason. Each probe runs only
# when an op asks for its flag, so listing ops never imports torch.

def _probe_gpu() -> bool:
    try:
        from .. import gpu as _gpu
    except ImportError:                              # pragma: no cover
        import gpu as _gpu
    return _gpu.available()


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _probe_pixels() -> bool:
    from . import archive
    return archive.settings()["pixels"]


FLAG_PROBES: dict[str, Callable[[], bool]] = {
    "gpu": _probe_gpu,
    "diffusers": lambda: _has_module("diffusers"),
    "modflows": lambda: _has_module("torch"),
    "pixels": _probe_pixels,
    "ui": lambda: False,
    "frozen": lambda: bool(getattr(sys, "frozen", False)),
}

FLAG_MEANING = {
    "gpu": "a graphics card this build can use",
    "diffusers": "the diffusers library (img2img, IP-Adapter, ControlNet, generate)",
    "modflows": "PyTorch, for the ModFlows colour model",
    "pixels": "the person's permission for thumbnails to leave the machine",
    "ui": "the web app being open",
    "frozen": "the installed build",
}


# ─────────────────────────── the context of a call ───────────────────────────

@dataclass
class Context:
    """
    Who is calling, and what they may see.

    `pixels` says whether this caller may receive pictures at all. For an agent
    that is the person's switch in agent.json, read fresh for every context. A
    person at a terminal or in the app is looking at their own pictures, so for
    them it is always true.

    `sheets` says whether ops that can illustrate a text result (a search, a
    survey) should spend the time drawing a contact sheet. An agent with pixels
    gets one, since it cannot see the files otherwise. The CLI draws them when
    asked with --sheets. When `sheets` is off, those ops put hex palettes and
    descriptions in the text instead, since that is all the reader will get.
    """
    actor: str = "cli"
    pixels: bool = True
    sheets: bool = False
    _flags: dict = field(default_factory=dict, repr=False)

    def has(self, flag: str) -> bool:
        if flag == "pixels":
            return self.pixels
        if flag not in self._flags:
            probe = FLAG_PROBES.get(flag)
            if probe is None:
                raise KeyError(f"unknown capability flag {flag!r}")
            try:
                self._flags[flag] = bool(probe())
            except Exception:
                self._flags[flag] = False
        return self._flags[flag]

    def flags(self) -> dict[str, bool]:
        return {f: self.has(f) for f in FLAG_PROBES}


def context(actor: str = "cli", *, sheets: bool | None = None) -> Context:
    """The context for one caller. Agents get the person's pixels setting, read now."""
    if actor not in ACTORS:
        raise ValueError(f"actor is one of {', '.join(ACTORS)}, got {actor!r}")
    if actor == "agent":
        pixels = _probe_pixels()
        return Context(actor, pixels=pixels, sheets=pixels if sheets is None else (sheets and pixels))
    return Context(actor, pixels=True, sheets=bool(sheets))


# ─────────────────────────── ops ───────────────────────────

@dataclass(frozen=True)
class Op:
    name: str
    kind: Literal["read", "write", "effect"]
    args: type[Args]
    result: type[Result]
    run: Callable[[Args, Context], Result]
    doc: str
    examples: tuple[str, ...] = ()
    needs: frozenset[str] = frozenset()
    actors: frozenset[str] = frozenset(ACTORS)

    @property
    def summary(self) -> str:
        """The doc's first sentence, for the op list."""
        first = " ".join(self.doc.split()).split(". ")[0]
        return first.rstrip(".") + "."

    @property
    def words(self) -> tuple[str, ...]:
        return tuple(self.name.split())


_OPS: dict[str, Op] = {}


def register(op: Op) -> Op:
    """
    Add an op to the vocabulary, refusing anything that would break the rule.

    A second op with the same name, an op without a doc, an args model that
    would silently ignore an unknown field, and an example that does not parse
    are all programming errors, so they raise at import time rather than
    surfacing as a confused caller later.
    """
    problems = []
    if not NAME_RE.match(op.name):
        problems.append("names are one to three lowercase words, noun first")
    if op.name in _OPS:
        problems.append("an op with this name already exists")
    if op.kind not in KINDS:
        problems.append(f"kind is one of {KINDS}")
    if op.kind == "write":
        # Writes run as transactions on state.Store, with per-actor undo. That
        # arrives with the board ops in P1. Until then nothing may write.
        problems.append("write ops need state.Store, which arrives in P1")
    if not op.doc or not op.doc.strip():
        problems.append("every op needs a doc")
    if not (isinstance(op.args, type) and issubclass(op.args, Args)):
        problems.append("args must subclass ops.Args")
    elif op.args.model_config.get("extra") != "forbid":
        problems.append("args models must set extra='forbid'")
    if not (isinstance(op.result, type) and issubclass(op.result, Result)):
        problems.append("result must subclass ops.Result")
    if not op.needs <= set(FLAG_PROBES):
        problems.append(f"unknown flags in needs: {sorted(op.needs - set(FLAG_PROBES))}")
    if not op.actors <= set(ACTORS):
        problems.append(f"unknown actors: {sorted(op.actors - set(ACTORS))}")
    if problems:
        raise TypeError(f"op {op.name!r}: " + "; ".join(problems))

    _OPS[op.name] = op
    from . import grammar
    for line in op.examples:
        try:
            parsed = grammar.parse_line(line)
            if parsed.op.name != op.name:
                raise Refusal(op.name, f"resolves to {parsed.op.name!r}")
            validate(op, parsed.args)
        except Refusal as e:
            del _OPS[op.name]
            raise TypeError(f"op {op.name!r}: example {line!r} does not parse: {e.reason}") from None
    return op


def op(name: str, kind: str, args: type[Args], result: type[Result], *, doc: str,
       examples: tuple[str, ...] = (), needs=(), actors=ACTORS):
    """Decorator form of register(): the decorated function is the op's body."""
    def wrap(fn):
        register(Op(name, kind, args, result, fn, textwrap.dedent(doc).strip(),
                    tuple(examples), frozenset(needs), frozenset(actors)))
        return fn
    return wrap


def get(name: str) -> Op | None:
    return _OPS.get(name)


def all_ops() -> list[Op]:
    """Every op, in the order it was registered, which is the order the op list shows."""
    return list(_OPS.values())


def unregister(name: str) -> None:
    """For tests that register a scratch op."""
    _OPS.pop(name, None)


def suggest(word: str, choices, prefix: str = "") -> str:
    """' Did you mean x?' or empty. `prefix` spells the choice as typed (--k)."""
    close = difflib.get_close_matches(word, list(choices), n=2, cutoff=0.6)
    if not close:
        return ""
    return " Did you mean " + " or ".join(f"`{prefix}{c}`" for c in close) + "?"


# ─────────────────────────── validation ───────────────────────────

def _flag(op: Op, field_name: str) -> str:
    """How a field is spelled on the command line, for messages."""
    info = op.args.model_fields.get(field_name)
    extra = (info.json_schema_extra or {}) if info else {}
    if isinstance(extra, dict) and "positional" in extra:
        return f"<{field_name}>"
    return "--" + field_name.replace("_", "-")


def _explain(op: Op, err: ValidationError) -> str:
    """Pydantic's complaints, rewritten in the terms the caller typed."""
    out = []
    for e in err.errors():
        loc = [str(x) for x in e.get("loc", ())]
        name = loc[0] if loc else ""
        where = _flag(op, name) if name in op.args.model_fields else (f"--{name}" if name else "the arguments")
        if len(loc) > 1:
            where += " (" + ".".join(loc[1:]) + ")"
        t = e.get("type", "")
        ctx = e.get("ctx") or {}
        got = e.get("input")
        info = op.args.model_fields.get(name)
        bounds = _bounds(info) if info is not None else {}
        if t == "missing":
            out.append(f"{where} is required.")
        elif t == "extra_forbidden":
            out.append(f"{where} is not an option of `{op.name}`."
                       + suggest(name.replace("_", "-"), [f.replace("_", "-") for f in op.args.model_fields], "--"))
        elif t in ("greater_than_equal", "less_than_equal", "greater_than", "less_than"):
            if "ge" in bounds and "le" in bounds:
                out.append(f"{where} must be between {bounds['ge']} and {bounds['le']}, got {got}.")
            else:
                out.append(f"{where} {e.get('msg', '').lower()}, got {got}.")
        elif t in ("too_short", "too_long"):
            lo, hi = bounds.get("min_length"), bounds.get("max_length")
            n = len(got) if hasattr(got, "__len__") else got
            if lo is not None and hi is not None:
                out.append(f"{where} takes {lo} to {hi} values, got {n}.")
            elif lo is not None:
                out.append(f"{where} takes at least {lo} value{'s' if lo != 1 else ''}, got {n}.")
            else:
                out.append(f"{where} takes at most {hi} values, got {n}.")
        elif t == "literal_error":
            out.append(f"{where} must be one of {ctx.get('expected', '?')}. Got {got!r}.")
        elif t in ("int_parsing", "int_from_float"):
            out.append(f"{where} must be a whole number, got {got!r}.")
        elif t == "float_parsing":
            out.append(f"{where} must be a number, got {got!r}.")
        elif t == "bool_parsing":
            out.append(f"{where} must be true or false, got {got!r}.")
        elif t == "value_error":
            out.append(f"{where}: {str(ctx.get('error', e.get('msg', '')))}")
        else:
            out.append(f"{where}: {e.get('msg', 'invalid')}.")
    return " ".join(out)


def _bounds(info) -> dict:
    """ge, le, min_length, max_length from a field's metadata."""
    out = {}
    for m in getattr(info, "metadata", []) or []:
        for attr in ("ge", "le", "gt", "lt", "min_length", "max_length"):
            v = getattr(m, attr, None)
            if v is not None:
                out[attr] = v
    return out


def validate(op: Op, args) -> Args:
    """Parse a dict into the op's args model, or refuse in the caller's terms."""
    if isinstance(args, op.args):
        return args
    if isinstance(args, BaseModel):
        args = args.model_dump()
    try:
        return op.args.model_validate(args or {})
    except ValidationError as e:
        from . import grammar
        raise Refusal(op.name, _explain(op, e), grammar.help_text(op)) from None


# ─────────────────────────── call ───────────────────────────

def call(name: str, args=None, actor: str = "cli", ctx: Context | None = None) -> Result:
    """
    The one entry point. Look the op up, check who is asking and what this
    Arcana can do, parse the arguments, run it.

    Every expected failure leaves as a Refusal carrying the op's help. A
    ValueError or FileNotFoundError from an op body is expected: it is how the
    bodies say "no dataset called nope" or "this dataset has no palette
    features", and the message is already written for the caller.
    """
    from . import grammar
    found = get(name)
    if found is None:
        raise Refusal(None, f"No op called `{name}`." + suggest(name, _OPS)
                      + " `op list` shows them all.")
    ctx = ctx or context(actor)
    if ctx.actor not in found.actors:
        raise Refusal(found.name, f"`{found.name}` is for the person. Ask them to do it.",
                      grammar.help_text(found))
    missing = [f for f in sorted(found.needs) if not ctx.has(f)]
    if missing:
        raise Refusal(found.name,
                      f"`{found.name}` needs " + ", ".join(FLAG_MEANING[f] for f in missing)
                      + ", which this Arcana does not have.", grammar.help_text(found))
    parsed = validate(found, args)
    try:
        return found.run(parsed, ctx)
    except Refusal as e:
        if not e.op:
            e.op = found.name
        if not e.help:
            e.help = grammar.help_text(found)
        raise
    except (ValueError, FileNotFoundError) as e:
        raise Refusal(found.name, str(e), grammar.help_text(found)) from e


# ─────────────────────────── batches ───────────────────────────

class Step(BaseModel):
    """One line of a batch and what became of it."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    line: str
    op: str | None = None
    result: SerializeAsAny[Result] | None = None
    refusal: dict | None = None
    help: bool = False                    # the line asked for help rather than running


class Batch(BaseModel):
    steps: list[Step]

    @property
    def ok(self) -> bool:
        return all(s.refusal is None for s in self.steps)

    @property
    def refused(self) -> Step | None:
        return next((s for s in self.steps if s.refusal is not None), None)


def exec_lines(lines, actor: str = "cli", ctx: Context | None = None) -> Batch:
    """
    Run text lines as one batch: each line one op, in order, stopping at the
    first refusal.

    A batch is meant to be atomic, so that a refused third line leaves no trace
    of the first two. Every op registered so far is a read or an effect that
    returns its output without recording it, so stopping is enough for now.
    When writes arrive (P1) the batch becomes one transaction and one undo step
    on state.Store, and a refusal rolls back what came before it.

    A line that ends in --help answers with the op's help and runs nothing.
    Blank lines and lines starting with # are skipped.
    """
    from . import grammar
    ctx = ctx or context(actor)
    steps: list[Step] = []
    stopped = False
    for raw in lines:
        line = (raw or "").strip()
        if not line or line.startswith("#"):
            continue
        if stopped:
            steps.append(Step(line=line))
            continue
        try:
            parsed = grammar.parse_line(line)
            if parsed.help:
                steps.append(Step(line=line, op=parsed.op.name, help=True,
                                  result=TextResult(body=grammar.help_text(parsed.op))))
                continue
            result = call(parsed.op.name, parsed.args, ctx=ctx)
            steps.append(Step(line=line, op=parsed.op.name, result=result))
        except Refusal as e:
            steps.append(Step(line=line, op=e.op, refusal=e.as_dict()))
            stopped = True
    return Batch(steps=steps)

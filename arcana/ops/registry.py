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

try:
    from .. import modality as _modality
except ImportError:                                  # pragma: no cover
    import modality as _modality

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


# ─────────────────────────── modalities ───────────────────────────
# What kind of media an op works with: photos, sounds, videos. An op that takes
# a dataset or pictures declares it (`modalities=`), `op list` and `--help`
# print it as a tag, and call() refuses media of another kind before the body
# runs, in one sentence that names what the op takes and what it was given.
# An op that takes no media (orient, undo, job list) declares nothing.
#
# A field holds media when it names a dataset (it completes as one) or when its
# metavar is one of the words below, which is how every op already spells a
# picture argument.

MEDIA_METAVARS = frozenset({"HANDLE", "ITEM", "PICTURE", "MATERIAL"})


def media_fields(args_model) -> list[tuple[str, str]]:
    """(field name, "dataset" or "item") for each field of an args model that holds media."""
    out = []
    for name, info in args_model.model_fields.items():
        extra = info.json_schema_extra if isinstance(info.json_schema_extra, dict) else {}
        if extra.get("complete") == "dataset":
            out.append((name, "dataset"))
        elif extra.get("metavar") in MEDIA_METAVARS:
            out.append((name, "item"))
    return out


def modality_of_ref(ref: str, role: str = "item") -> str | None:
    """
    The modality a reference names, or None when it names no media this can tell.

    A dataset id carries it in its suffix, and so does a handle. A path and a
    derived picture (out:3f9c1e) are read from the file's extension. A board
    number (#3), an empty value and anything unrecognised answer None, and the
    op's body decides.
    """
    ref = (ref or "").strip()
    if not ref:
        return None
    if role == "dataset":
        return _modality.split_dataset_id(ref)[1]
    if "/" in ref or "\\" in ref:
        return _modality.of_path(ref)
    if ref.startswith("out:"):
        try:
            from .. import ledger as _ledger
        except ImportError:                          # pragma: no cover
            import ledger as _ledger
        try:
            entry = _ledger.get(ref)
        except Exception:
            entry = None
        return _modality.of_path(entry["path"]) if entry and entry.get("path") else None
    return _modality.of_handle(ref)


def check_modalities(op: "Op", parsed: Args) -> None:
    """Refuse media of a kind the op does not take. Runs before the op's body."""
    if op.modalities is None:
        return
    for name, role in media_fields(op.args):
        value = getattr(parsed, name, None)
        refs = value if isinstance(value, (list, tuple)) else [value]
        for ref in refs:
            if not isinstance(ref, str):
                continue
            got = modality_of_ref(ref, role)
            if got is None or got in op.modalities:
                continue
            if role == "dataset":
                what = f"{ref} holds {_modality.PLURAL[got]}"
            else:
                what = f"{ref} is a {_modality.NOUN[got]}"
            raise Refusal(op.name, f"`{op.name}` works with {_modality.words(op.modalities)}. "
                                   f"{what}.")


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
    tx: object = None             # the open state.Transaction, for write and effect ops
    progress: object = None       # progress(fraction, message), when a job or the UI listens
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
    # False for undo and redo, which move history entries themselves and must
    # not be recorded as steps of their own.
    journal: bool = True
    # The kinds of media the op works with, or None for an op that takes none.
    modalities: frozenset[str] | None = None

    @property
    def tags(self) -> list[str]:
        """The modalities in display order, ["image", "audio"]. Empty for an op that takes no media."""
        return _modality.tags(self.modalities or ())

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
    if op.modalities is not None and (not op.modalities or not op.modalities <= set(_modality.ALL)):
        problems.append(f"modalities are some of {_modality.ALL}, got {sorted(op.modalities)}")
    if (op.modalities is None and isinstance(op.args, type) and issubclass(op.args, Args)
            and media_fields(op.args)):
        problems.append("an op that takes a dataset or pictures declares its modalities ("
                        + ", ".join(n for n, _ in media_fields(op.args)) + ")")
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
       examples: tuple[str, ...] = (), needs=(), actors=ACTORS, journal: bool = True,
       modalities=None):
    """Decorator form of register(): the decorated function is the op's body."""
    def wrap(fn):
        register(Op(name, kind, args, result, fn, textwrap.dedent(doc).strip(),
                    tuple(examples), frozenset(needs), frozenset(actors), journal,
                    None if modalities is None else frozenset(modalities)))
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

def _store():
    try:
        from .. import state as _state
    except ImportError:                              # pragma: no cover
        import state as _state
    return _state


def call(name: str, args=None, actor: str = "cli", ctx: Context | None = None,
         label: str | None = None) -> Result:
    """
    The one entry point. Look the op up, check who is asking and what this
    Arcana can do, parse the arguments, run it.

    A write or an effect runs inside a transaction on state.STORE: committed as
    one undo step for this actor when the op returns, rolled back when it
    refuses. Inside a batch, the batch's transaction is used instead.

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
        check_modalities(found, parsed)
    except Refusal as e:
        e.help = grammar.help_text(found)
        raise
    st = _store()
    if not found.journal and ctx.tx is not None:
        raise Refusal(found.name, f"`{found.name}` runs on its own, outside a batch that "
                                  f"changes anything.", grammar.help_text(found))
    own_tx = found.kind != "read" and found.journal and ctx.tx is None
    if own_tx:
        ctx.tx = st.STORE.begin(ctx.actor, label or found.name, lazy=found.kind == "effect")
    try:
        try:
            result = found.run(parsed, ctx)
        except Refusal as e:
            if not e.op:
                e.op = found.name
            if not e.help:
                e.help = grammar.help_text(found)
            raise
        except (ValueError, FileNotFoundError) as e:
            # state.StateError is a ValueError: A7 and stale undos land here.
            raise Refusal(found.name, str(e), grammar.help_text(found)) from e
        if own_tx:
            tx, ctx.tx = ctx.tx, None
            st.STORE.commit(tx)
        _note(ctx.actor, found, parsed, result)
        return result
    except BaseException:
        if own_tx and ctx.tx is not None:
            tx, ctx.tx = ctx.tx, None
            st.STORE.rollback(tx)
        raise


def _note(actor: str, op: Op, parsed: Args, result: Result) -> None:
    """Tell the page what just ran (activity.py). Never allowed to fail the op."""
    try:
        try:
            from .. import activity as _activity
        except ImportError:                          # pragma: no cover
            import activity as _activity
        try:
            text = result.text()
        except Exception:
            text = ""
        small = {k: v for k, v in result.model_dump(mode="json", exclude={"attachments"}).items()
                 if isinstance(v, (str, int, float, bool)) and (not isinstance(v, str) or len(v) < 400)}
        _activity.note(actor, op.name, parsed.model_dump(mode="json", exclude_defaults=True), text, small)
    except Exception:
        pass


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
    rolled_back: bool = False             # a refusal undid the writes before it
    version: int | None = None            # state version after a batch that wrote

    @property
    def ok(self) -> bool:
        return all(s.refusal is None for s in self.steps)

    @property
    def refused(self) -> Step | None:
        return next((s for s in self.steps if s.refusal is not None), None)


def exec_lines(lines, actor: str = "cli", ctx: Context | None = None) -> Batch:
    """
    Run text lines as one batch: each line one op, in order.

    Every line is parsed before any runs, so a typo on line five stops the
    batch before line one changes anything. If the batch writes, it runs as one
    transaction and one undo step, and a refusal on any line rolls back every
    change the lines before it made. Reads run in order with the writes around
    them, so a batch can add pictures to a board and then show the board.

    A line that ends in --help answers with the op's help and runs nothing.
    Blank lines and lines starting with # are skipped.
    """
    from . import grammar
    ctx = ctx or context(actor)
    wanted = []
    for raw in lines:
        line = (raw or "").strip()
        if line and not line.startswith("#"):
            wanted.append(line)

    parsed = []
    for i, line in enumerate(wanted):
        try:
            parsed.append(grammar.parse_line(line))
        except Refusal as e:
            steps = [Step(line=l) for l in wanted]
            steps[i] = Step(line=line, op=e.op, refusal=e.as_dict())
            return Batch(steps=steps)

    changing = [p for p in parsed if p.op.kind != "read" and p.op.journal and not p.help]
    st = _store()
    own_tx = bool(changing) and ctx.tx is None
    if own_tx:
        label = "; ".join(wanted)
        only_effects = all(p.op.kind == "effect" for p in changing)
        ctx.tx = st.STORE.begin(ctx.actor, label if len(label) <= 160 else label[:157] + "...",
                                lazy=only_effects)
    steps: list[Step] = []
    refused = False
    try:
        for line, p in zip(wanted, parsed):
            if refused:
                steps.append(Step(line=line))
                continue
            if p.help:
                steps.append(Step(line=line, op=p.op.name, help=True,
                                  result=TextResult(body=grammar.help_text(p.op))))
                continue
            try:
                result = call(p.op.name, p.args, ctx=ctx)
                steps.append(Step(line=line, op=p.op.name, result=result))
            except Refusal as e:
                steps.append(Step(line=line, op=e.op, refusal=e.as_dict()))
                refused = True
    except BaseException:
        if own_tx:
            tx, ctx.tx = ctx.tx, None
            st.STORE.rollback(tx)
        raise
    batch = Batch(steps=steps)
    if own_tx:
        tx, ctx.tx = ctx.tx, None
        if refused:
            st.STORE.rollback(tx)
            batch.rolled_back = bool(tx.before)
        else:
            batch.version = st.STORE.commit(tx)
    return batch

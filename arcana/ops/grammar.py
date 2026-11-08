# grammar.py -- ops as text lines
#
# One line is one op. The op name is the longest registered prefix of the line,
# so "search text japan --phrase rain" resolves to `search text`. The rest maps
# onto the op's args model by rules read from the model's fields:
#
#   scalar             --k 12, --k=12
#   bool               --full-res, --no-full-res
#   list of scalars    the flag repeated: --phrase "rain" --phrase "wet asphalt"
#   Literal            --mode palette, choices in --help and in completion
#   positional         bare words, in order; a trailing list takes the rest
#   nested model       JSON in one argument: --facet '{"name": "w", "phrases": ["rain"]}'
#   the whole model    --args '{"dataset": "japan", ...}' or --args @request.json
#
# Flag names are field names in kebab case, so a refusal that names --k names
# the same thing in the help, the JSON Schema and the Python model.
#
# The CLI and the MCP server share this, so a person can paste an agent's line
# into a terminal and get the same answer.

from __future__ import annotations

import json
import shlex
import textwrap
import types
import typing
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple, get_args, get_origin

from pydantic import BaseModel

from . import registry
from .registry import Op, Refusal

WIDTH = 78
COL = 24          # where descriptions start in the help's option table


# ─────────────────────────── splitting ───────────────────────────

def split(line: str) -> list[str]:
    """
    Shell-like words, with two departures from POSIX that Windows needs.

    Backslash is an ordinary character, so C:\\Users\\me\\Pictures arrives
    whole, quoted or not. And # starts nothing, since it starts every hex
    colour. Quote with either kind of quote; to put one kind inside a word,
    wrap the word in the other.
    """
    lex = shlex.shlex(line, posix=True)
    lex.whitespace_split = True
    lex.commenters = ""
    lex.escape = ""
    try:
        return list(lex)
    except ValueError as e:                       # "No closing quotation"
        raise Refusal(None, f"Could not read the line: {str(e).lower()}.") from None


def join(tokens) -> str:
    """The inverse of split(): a line that splits back into exactly `tokens`."""
    out = []
    for t in tokens:
        t = str(t)
        if t and not any(c in t for c in " \t\n'\""):
            out.append(t)
        elif "'" not in t:
            out.append(f"'{t}'")
        elif '"' not in t:
            out.append(f'"{t}"')
        else:                                     # both quotes: concatenate pieces
            out.append("'" + t.replace("'", "'\"'\"'") + "'")
    return " ".join(out)


# ─────────────────────────── fields ───────────────────────────

@dataclass(frozen=True)
class Spec:
    """What the grammar needs to know about one field of an args model."""
    name: str
    flag: str                      # without the leading dashes
    shape: str                     # scalar, bool, list, model, model_list
    base: Any                      # the element type
    positional: int | None
    required: bool
    default: Any
    choices: tuple | None
    description: str
    bounds: dict
    complete: str | None           # "dataset", "handle", ... for completion
    metavar: str


def _unwrap_optional(t):
    origin = get_origin(t)
    if origin in (typing.Union, types.UnionType):
        rest = [a for a in get_args(t) if a is not type(None)]
        if len(rest) == 1:
            return rest[0]
    return t


def _is_model(t) -> bool:
    return isinstance(t, type) and issubclass(t, BaseModel)


def _field_order(model) -> list[str]:
    """A model's own fields first, then those it inherits, so shared filters come last."""
    order = []
    for cls in model.__mro__:
        for name in getattr(cls, "__annotations__", {}):
            if name in model.model_fields and name not in order:
                order.append(name)
    return order + [n for n in model.model_fields if n not in order]


def specs(op: Op) -> list[Spec]:
    out = []
    for name in _field_order(op.args):
        info = op.args.model_fields[name]
        t = _unwrap_optional(info.annotation)
        origin = get_origin(t)
        if t is bool:
            shape, base = "bool", bool
        elif origin in (list, tuple, set):
            inner = _unwrap_optional((get_args(t) or (str,))[0])
            shape, base = ("model_list" if _is_model(inner) else "list"), inner
        elif _is_model(t):
            shape, base = "model", t
        else:
            shape, base = "scalar", t
        choices = tuple(get_args(base)) if get_origin(base) is Literal else None
        extra = info.json_schema_extra if isinstance(info.json_schema_extra, dict) else {}
        required = info.is_required()
        default = None if required else info.get_default(call_default_factory=True)
        if choices:
            metavar = "|".join(str(c) for c in choices)
        elif shape in ("model", "model_list"):
            metavar = "JSON"
        else:
            metavar = extra.get("metavar") or {int: "N", float: "X"}.get(base, "TEXT")
        out.append(Spec(name=name, flag=name.replace("_", "-"), shape=shape, base=base,
                        positional=extra.get("positional"), required=required,
                        default=default, choices=choices,
                        description=(info.description or "").strip(),
                        bounds=registry._bounds(info), complete=extra.get("complete"),
                        metavar=metavar))
    return out


# ─────────────────────────── resolving the op ───────────────────────────

def resolve(tokens: list[str]) -> tuple[Op, list[str]]:
    """The op named by the longest registered prefix of `tokens`, and the rest."""
    for n in (3, 2, 1):
        if len(tokens) >= n:
            found = registry.get(" ".join(tokens[:n]))
            if found is not None:
                return found, tokens[n:]
    words = [t for t in tokens[:3] if not t.startswith("-")]
    if not words:
        raise Refusal(None, "An op line starts with the op's name. `op list` shows them all.")
    names = [o.name for o in registry.all_ops()]
    nouns = sorted({n.split()[0] for n in names})
    if words[0] in nouns:
        verbs = [n for n in names if n.split()[0] == words[0]]
        tried = " ".join(words[:2])
        raise Refusal(None, f"`{tried}` is not an op. The `{words[0]}` ops are "
                      + ", ".join(f"`{v}`" for v in verbs) + "."
                      + registry.suggest(tried, verbs))
    tried = " ".join(words[:2])
    raise Refusal(None, f"No op called `{words[0]}`."
                  + (registry.suggest(tried, names) or registry.suggest(words[0], nouns))
                  + " `op list` shows them all.")


# ─────────────────────────── parsing ───────────────────────────

class Parsed(NamedTuple):
    op: Op
    args: dict
    help: bool = False


def _json_arg(op: Op, flag: str, raw: str):
    if raw.startswith("@"):
        try:
            with open(raw[1:], encoding="utf-8") as fh:
                raw = fh.read()
        except OSError as e:
            raise Refusal(op.name, f"`--{flag}` could not read {raw[1:]!r}: {e.strerror}.",
                          help_text(op)) from None
    try:
        return json.loads(raw)
    except ValueError as e:
        raise Refusal(op.name, f"`--{flag}` takes JSON, and {raw[:60]!r} is not JSON ({e}).",
                      help_text(op)) from None


def _bool(op: Op, flag: str, raw: str) -> bool:
    low = raw.strip().lower()
    if low in ("1", "true", "yes", "on"):
        return True
    if low in ("0", "false", "no", "off"):
        return False
    raise Refusal(op.name, f"`--{flag}` is on or off, got {raw!r}.", help_text(op))


def _coerce(spec: Spec, raw: str):
    """
    A word from the line, as the field's choices expect it.

    Pydantic coerces "12" to an int field, but leaves "3" unequal to a choice
    of the integer 3, so integer choices (--dims 2|3) are converted here.
    """
    if spec.choices and all(isinstance(c, int) and not isinstance(c, bool) for c in spec.choices):
        try:
            return int(raw)
        except ValueError:
            return raw
    return raw


def parse_tokens(tokens: list[str]) -> Parsed:
    """Words (already split, as the CLI receives them) into an op and its args dict."""
    op, rest = resolve(list(tokens))
    fields = specs(op)
    by_flag = {s.flag: s for s in fields}
    values: dict = {}
    bare: list[str] = []
    whole: dict | None = None

    def refuse(reason):
        raise Refusal(op.name, reason, help_text(op))

    i = 0
    only_bare = False
    while i < len(rest):
        t = rest[i]
        i += 1
        if only_bare or not t.startswith("--") or t == "--":
            if t == "--" and not only_bare:
                only_bare = True
                continue
            if not only_bare and t in ("-h",):
                return Parsed(op, {}, True)
            bare.append(t)
            continue
        name, eq, val = t[2:].partition("=")
        if name == "help":
            return Parsed(op, {}, True)
        if name == "args":
            if not eq:
                if i >= len(rest):
                    refuse("`--args` needs a JSON object or @file.json.")
                val, i = rest[i], i + 1
            got = _json_arg(op, "args", val)
            if not isinstance(got, dict):
                refuse("`--args` takes a JSON object, with the op's fields as keys.")
            whole = got
            continue
        spec = by_flag.get(name)
        if spec is None and name.startswith("no-") and by_flag.get(name[3:]) is not None \
                and by_flag[name[3:]].shape == "bool":
            if eq:
                refuse(f"`--{name}` takes no value.")
            values[by_flag[name[3:]].name] = False
            continue
        if spec is None:
            refuse(f"`--{name}` is not an option of `{op.name}`."
                   + registry.suggest(name, by_flag, "--"))
        if spec.shape == "bool":
            values[spec.name] = _bool(op, name, val) if eq else True
            continue
        if not eq:
            if i >= len(rest):
                refuse(f"`--{name}` needs a value ({spec.metavar}).")
            val, i = rest[i], i + 1
        if spec.shape == "list":
            values.setdefault(spec.name, []).append(_coerce(spec, val))
        elif spec.shape == "model_list":
            got = _json_arg(op, name, val)
            values.setdefault(spec.name, []).extend(got if isinstance(got, list) else [got])
        elif spec.shape == "model":
            values[spec.name] = _json_arg(op, name, val)
        else:
            if spec.name in values:
                refuse(f"`--{name}` was given twice.")
            values[spec.name] = _coerce(spec, val)

    # Bare words fill the positional fields in order; a list takes the rest.
    slots = sorted((s for s in fields if s.positional is not None), key=lambda s: s.positional)
    for s in slots:
        if not bare:
            break
        if s.name in values:
            refuse(f"<{s.name}> was given both bare and as `--{s.flag}`.")
        if s.shape in ("list", "model_list"):
            values[s.name] = bare if s.shape == "list" else [_json_arg(op, s.flag, b) for b in bare]
            bare = []
        else:
            values[s.name] = bare.pop(0)
    if bare:
        refuse(f"Unexpected {join(bare)!s}. `{op.name}` takes "
               + (", ".join(f"<{s.name}>" for s in slots) + " bare" if slots else "no bare words")
               + ", and everything else as --flags.")

    if whole is not None:
        clash = sorted(set(whole) & set(values))
        if clash:
            refuse("Given both in --args and on the line: " + ", ".join(clash) + ".")
        values = {**whole, **values}
    return Parsed(op, values, False)


def parse_line(line: str) -> Parsed:
    return parse_tokens(split(line))


# ─────────────────────────── help ───────────────────────────

def _default_text(s: Spec) -> str:
    if s.required or s.default in (None, [], ()):
        return ""
    if s.shape == "bool":
        return "on" if s.default else ""
    return f"default {s.default}"


def _constraint_text(s: Spec) -> str:
    b = s.bounds
    bits = []
    if "ge" in b and "le" in b:
        bits.append(f"{b['ge']} to {b['le']}")
    elif "ge" in b:
        bits.append(f"at least {b['ge']}")
    elif "le" in b:
        bits.append(f"at most {b['le']}")
    if s.shape in ("list", "model_list"):
        lo, hi = b.get("min_length"), b.get("max_length")
        if s.positional is None:
            bits.append("repeatable")
        if lo is not None and hi is not None:
            bits.append(f"{lo} to {hi} values")
        elif hi is not None:
            bits.append(f"up to {hi}")
    return ", ".join(bits)


def _term(s: Spec) -> str:
    if s.positional is not None:
        return f"<{s.name}>" + ("..." if s.shape in ("list", "model_list") else "")
    if s.shape == "bool":
        return f"--{s.flag}, --no-{s.flag}"
    return f"--{s.flag} {s.metavar}"


def usage_terms(op: Op) -> list[str]:
    parts = [op.name]
    fields = specs(op)
    for s in sorted((s for s in fields if s.positional is not None), key=lambda s: s.positional):
        term = _term(s)
        parts.append(term if s.required else f"[{term}]")
    for s in fields:
        if s.positional is not None:
            continue
        term = f"--{s.flag}" if s.shape == "bool" else f"--{s.flag} {s.metavar}"
        if s.shape in ("list", "model_list"):
            term += " ..."
        parts.append(term if s.required else f"[{term}]")
    return parts


def usage(op: Op) -> str:
    return " ".join(usage_terms(op))


def _wrap_terms(terms: list[str], width: int, indent: str = "    ") -> list[str]:
    """Wrap a usage line between terms, never inside one."""
    lines, cur = [], ""
    for t in terms:
        if cur and len(cur) + 1 + len(t) > width:
            lines.append(cur)
            cur = indent + t
        else:
            cur = f"{cur} {t}" if cur else t
    return lines + [cur]


def help_text(op: Op) -> str:
    """The op's help page, generated from its declaration."""
    lines = _wrap_terms(usage_terms(op), WIDTH)
    lines.append(f"({op.kind}" + (f", needs {', '.join(sorted(op.needs))}" if op.needs else "")
                 + (f", {' and '.join(a for a in registry.ACTORS if a in op.actors)} only"
                    if op.actors != frozenset(registry.ACTORS) else "") + ")")
    lines.append("")
    for para in op.doc.split("\n\n"):
        lines += textwrap.wrap(" ".join(para.split()), WIDTH)
        lines.append("")
    fields = specs(op)
    for title, group in (("Arguments", [s for s in fields if s.positional is not None]),
                         ("Options", [s for s in fields if s.positional is None])):
        if not group:
            continue
        lines.append(title)
        for s in group:
            term = "  " + _term(s)
            notes = [n for n in (("required" if s.required and s.positional is None else ""),
                                 _constraint_text(s), _default_text(s)) if n]
            desc = (s.description + (" " if s.description else "")
                    + (f"({', '.join(notes)})" if notes else "")).strip()
            body = textwrap.wrap(desc, WIDTH - COL) or [""]
            if len(term) >= COL - 1:
                lines.append(term)
                lines += [" " * COL + b for b in body]
            else:
                lines.append(term.ljust(COL) + body[0])
                lines += [" " * COL + b for b in body[1:]]
        lines.append("")
    if op.examples:
        lines.append("Examples")
        lines += [f"  {e}" for e in op.examples]
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


GRAMMAR = """\
One line is one op: its name, then bare arguments, then --flags.

  search text japan --phrase "rain on a dark street" --phrase "wet asphalt" --k 8
  search like japan:1423 japan:88 --mode palette
  item show japan:1423

Lists repeat the flag (--phrase a --phrase b). On or off flags take no value
(--full-res, --no-full-res). Nested values are JSON in one argument, quoted
with single quotes (--facet '{"name": "w", "phrases": ["rain"]}'). The whole
argument set can come as --args '{...}' or --args @file.json.

Pictures are named by handle, dataset:key (japan:1423). Datasets are named by
id (japan, or field@audio for an audio set). Backslashes in Windows paths need
no escaping, and # needs no quoting.

<op> --help prints any op's help. `op list` lists them all.
"""


# ─────────────────────────── completion ───────────────────────────

def _dataset_ids() -> list[str]:
    from . import archive
    try:
        from .. import core
    except ImportError:                              # pragma: no cover
        import core
    return [archive.dataset_id(d["name"], d["modality"]) for d in core.find_datasets()]


def _board_ids() -> list[str]:
    try:
        from .. import boards as _boards
    except ImportError:                              # pragma: no cover
        import boards as _boards
    return [b["slug"] for b in _boards.listing()]


_COMPLETERS = {"dataset": _dataset_ids, "board": _board_ids}


def complete(partial: str) -> list[str]:
    """
    Candidates for the word being typed at the end of `partial`.

    A trailing space means a new word has begun. Op names complete word by
    word, flags complete after --, choices complete after their flag, and a
    dataset argument completes from the datasets on disk.
    """
    try:
        words = split(partial)
    except Refusal:
        return []
    if words and words[0] in ("arcana", "arcana.exe"):
        words = words[1:]
    current = "" if (not partial or partial[-1].isspace()) else (words.pop() if words else "")
    names = [o.name.split() for o in registry.all_ops()]

    # Still naming the op.
    resolved = None
    for n in (3, 2, 1):
        if len(words) >= n and registry.get(" ".join(words[:n])):
            resolved = registry.get(" ".join(words[:n]))
            rest = words[n:]
            break
    if resolved is None:
        depth = len(words)
        cands = {w[depth] for w in names
                 if len(w) > depth and w[:depth] == words and w[depth].startswith(current)}
        return sorted(cands)

    fields = specs(resolved)
    by_flag = {s.flag: s for s in fields}
    if rest:
        prev = rest[-1]
        if prev.startswith("--") and prev[2:] in by_flag:
            s = by_flag[prev[2:]]
            if s.shape != "bool":
                if s.choices:
                    return [str(c) for c in s.choices if str(c).startswith(current)]
                if s.complete in _COMPLETERS:
                    return [d for d in _COMPLETERS[s.complete]() if d.startswith(current)]
                return []
    if current.startswith("-"):
        flags = []
        for s in fields:
            flags.append(f"--{s.flag}")
            if s.shape == "bool":
                flags.append(f"--no-{s.flag}")
        flags += ["--help", "--args"]
        return sorted(f for f in flags if f.startswith(current))
    # A bare word: the next positional, if it completes.
    used = sum(1 for w in rest if not w.startswith("--"))
    slots = sorted((s for s in fields if s.positional is not None), key=lambda s: s.positional)
    if used < len(slots) and slots[used].complete in _COMPLETERS:
        return [d for d in _COMPLETERS[slots[used].complete]() if d.startswith(current)]
    return []


_COMPLETION = {
    "powershell": r"""# Arcana completion for PowerShell. Add to your profile:
#   arcana completion powershell | Out-String | Invoke-Expression
Register-ArgumentCompleter -Native -CommandName arcana, arcana.exe -ScriptBlock {
    param($wordToComplete, $commandAst, $cursorPosition)
    $text = $commandAst.Extent.Text
    $upto = [Math]::Min($text.Length, $cursorPosition - $commandAst.Extent.StartOffset)
    $line = $text.Substring(0, $upto)
    if ($wordToComplete -eq '' -and -not $line.EndsWith(' ')) { $line += ' ' }
    arcana complete $line | ForEach-Object {
        [System.Management.Automation.CompletionResult]::new($_, $_, 'ParameterValue', $_)
    }
}
""",
    "bash": r"""# Arcana completion for bash. Add to ~/.bashrc:
#   eval "$(arcana completion bash)"
_arcana_complete() {
    local IFS=$'\n'
    COMPREPLY=( $(arcana complete "${COMP_LINE:0:$COMP_POINT}") )
}
complete -o default -F _arcana_complete arcana
""",
    "zsh": r"""# Arcana completion for zsh. Add to ~/.zshrc:
#   eval "$(arcana completion zsh)"
autoload -U +X bashcompinit && bashcompinit
_arcana_complete() {
    local IFS=$'\n'
    COMPREPLY=( $(arcana complete "${COMP_LINE:0:$COMP_POINT}") )
}
complete -o default -F _arcana_complete arcana
""",
}


def completion_script(shell: str) -> str:
    try:
        return _COMPLETION[shell]
    except KeyError:
        raise Refusal(None, f"Completion scripts exist for {', '.join(_COMPLETION)}. "
                            f"Got {shell!r}.") from None


# ─────────────────────────── schema ───────────────────────────

def schema() -> dict:
    """Every op as JSON Schema, for clients that build calls from a schema."""
    return {o.name: {"kind": o.kind, "summary": o.summary, "doc": o.doc,
                     "needs": sorted(o.needs), "actors": sorted(o.actors),
                     "examples": list(o.examples),
                     "args": o.args.model_json_schema(),
                     "result": o.result.model_json_schema()}
            for o in registry.all_ops()}

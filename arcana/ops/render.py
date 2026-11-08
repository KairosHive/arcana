# render.py -- results as text, as JSON, and as files
#
# Each op's result model renders its own text. This module puts results and
# batches together for a surface, writes attachments to disk for the CLI, and
# applies the pixels gate for agents in one place.

from __future__ import annotations

import os
import re
import time

from .registry import Attachment, Batch, Context, Result

try:
    from .. import paths as _paths
except ImportError:                                   # pragma: no cover
    import paths as _paths


def refusal_text(refusal: dict, with_help: bool = True) -> str:
    out = f"Refused: {refusal['reason']}"
    if with_help and refusal.get("help"):
        out += "\n\n" + refusal["help"].rstrip()
    return out + "\n"


def result_json(result: Result) -> dict:
    """The result's fields, with attachments as captions and paths (never bytes)."""
    return result.model_dump(mode="json")


def visible_attachments(result: Result, ctx: Context) -> list[Attachment]:
    """
    The pictures this caller may receive.

    Ops already skip drawing what a blind agent cannot have. This is the second
    line: whatever an op attached, an agent without the person's permission
    receives none of it.
    """
    if ctx.actor == "agent" and not ctx.pixels:
        return []
    return list(result.attachments)


def batch_text(batch: Batch) -> str:
    """A batch as one page: each line's answer, the refusal with help, then what did not run."""
    parts = []
    many = len(batch.steps) > 1
    for i, s in enumerate(batch.steps, 1):
        head = f"[{i}] {s.line}\n" if many else ""
        if s.result is not None:
            # A picture the CLI wrote to disk is named under its own line.
            files = "".join(f"Sheet: {a.path}\n" for a in s.result.attachments if a.path)
            parts.append(head + s.result.text().rstrip() + "\n" + files)
        elif s.refusal is not None:
            ran_before = any(x.result is not None for x in batch.steps[:i - 1])
            if not many:
                where = ""
            elif not ran_before:
                where = f"Line {i} was refused, so nothing in the batch ran.\n"
            else:
                where = (f"Line {i} was refused, so nothing after it ran"
                         + (" and the changes before it were undone" if batch.rolled_back else "")
                         + ".\n")
            parts.append(head + where + refusal_text(s.refusal))
        else:
            parts.append(head + "(not run)\n")
    return "\n".join(parts)


def batch_json(batch: Batch) -> dict:
    return {
        "ok": batch.ok,
        "steps": [{"line": s.line, "op": s.op, "help": s.help,
                   "result": result_json(s.result) if s.result is not None else None,
                   "refusal": s.refusal,
                   "ran": s.result is not None}
                  for s in batch.steps],
    }


def sheets_dir() -> str:
    return _paths.ensure_dir(os.path.join(_paths.subdir("output"), "sheets"))


def write_attachments(result: Result, op_name: str = "sheet", folder: str | None = None) -> list[str]:
    """
    Write a result's pictures to files and record where, for the CLI.

    Names carry the op and the time, so sheets from a session sort in the order
    they were made and never overwrite one another.
    """
    if not result.attachments:
        return []
    folder = folder or sheets_dir()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    slug = re.sub(r"[^a-z0-9]+", "-", op_name.lower()).strip("-") or "sheet"
    written = []
    for i, att in enumerate(result.attachments, 1):
        ext = ".png" if att.mime == "image/png" else ".jpg"
        path = os.path.join(folder, f"{stamp}-{slug}-{i}{ext}")
        n = 2
        while os.path.exists(path):
            path = os.path.join(folder, f"{stamp}-{slug}-{i}-{n}{ext}")
            n += 1
        with open(path, "wb") as fh:
            fh.write(att.data)
        att.path = path
        written.append(path)
    return written

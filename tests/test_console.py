"""
A print statement must never be able to destroy hours of work.

On Windows sys.stdout defaults to the ANSI code page (cp1252 on the machine
this was written on), and printing anything outside it raises
UnicodeEncodeError. Nothing catches that -- nobody wraps a progress message in
a try block -- so it propagates out of whatever was running and kills it.

It did exactly that: indexing 61,039 photographs with ViT-H/14 reached the last
step of style extraction, printed "41152 -> 512 dims" with a real arrow in it,
and died. The CLIP pass and the palette pass were already saved; the style pass
and the whole layout were lost.

Run with:  python -m pytest tests/test_console.py -q
"""

import io
import os
import pathlib
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arcana import paths  # noqa: E402

PACKAGE = pathlib.Path(__file__).resolve().parent.parent / "arcana"

# The narrowest code page any of this is likely to meet. If a string survives
# cp1252 it survives cp437, latin-1 and every UTF variant too.
NARROW = "cp1252"


def _unencodable(text: str) -> list[str]:
    out = []
    for ch in text:
        if ord(ch) < 128:
            continue
        try:
            ch.encode(NARROW)
        except UnicodeEncodeError:
            out.append(ch)
    return out


def test_no_print_carries_a_character_the_console_cannot_take():
    """
    The second line of defence, and the reason it exists.

    use_utf8_console() is the first, but it cannot help a caller who imports
    arcana as a library and never runs an entry point, and a frozen build can
    be launched with no console at all. Keeping the strings themselves ASCII
    costs nothing -- "->" reads as well as an arrow in a log.
    """
    offenders = []
    for path in sorted(PACKAGE.glob("*.py")):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "print(" not in line:
                continue
            bad = _unencodable(line)
            if bad:
                offenders.append(f"{path.name}:{lineno} {bad!r}  {line.strip()[:70]}")
    assert not offenders, (
        "print() lines carry characters cp1252 cannot encode:\n  "
        + "\n  ".join(offenders))


def test_use_utf8_console_switches_the_encoding(monkeypatch):
    class Fake(io.StringIO):
        encoding = NARROW

        def __init__(self):
            super().__init__()
            self.calls = []

        def reconfigure(self, **kw):
            self.calls.append(kw)
            self.encoding = kw.get("encoding", self.encoding)

    out, err = Fake(), Fake()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    paths.use_utf8_console()

    for stream in (out, err):
        assert stream.calls == [{"encoding": "utf-8", "errors": "replace"}]
        # errors="replace", not strict: a filename from somewhere we do not
        # control should degrade to "?" rather than raise.
        assert stream.calls[0]["errors"] == "replace"


def test_a_frozen_build_with_no_console_does_not_crash(monkeypatch):
    """pythonw and a windowed PyInstaller build both set sys.stdout to None."""
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)
    paths.use_utf8_console()          # must simply do nothing


def test_a_stream_that_refuses_is_not_fatal(monkeypatch):
    class Stubborn:
        encoding = NARROW

        def reconfigure(self, **kw):
            raise ValueError("underlying buffer has been detached")

    monkeypatch.setattr(sys, "stdout", Stubborn())
    monkeypatch.setattr(sys, "stderr", Stubborn())
    paths.use_utf8_console()          # a convenience, never a reason to refuse to start


def test_calling_it_twice_is_harmless(monkeypatch):
    calls = []

    class Fake:
        encoding = NARROW

        def reconfigure(self, **kw):
            calls.append(kw)

    monkeypatch.setattr(sys, "stdout", Fake())
    monkeypatch.setattr(sys, "stderr", Fake())
    paths.use_utf8_console()
    paths.use_utf8_console()
    assert len(calls) == 4                       # two streams, twice, no error


def test_every_entry_point_turns_the_console_utf8():
    """
    One missed entry point is one command that can still die on a log line.

    Checked by reading the source rather than by running them, because running
    them means loading torch and, for two of them, indexing something.
    """
    expected = {
        "arcana.py": ["main"],
        "db.py": ["main", "extend_main", "marks_main"],
        "legacy.py": ["main"],
        "relocate.py": ["main"],
    }
    missing = []
    for filename, functions in expected.items():
        source = (PACKAGE / filename).read_text(encoding="utf-8")
        for fn in functions:
            body = source.split(f"\ndef {fn}(", 1)
            assert len(body) == 2, f"{filename} has no {fn}()"
            # The call must come early -- before the work, not after it.
            head = body[1][:600]
            if "use_utf8_console()" not in head:
                missing.append(f"{filename}:{fn}")
    assert not missing, f"entry points that never fix the console: {missing}"


def main():
    print("This suite uses pytest's monkeypatch; run it with:")
    print("    python -m pytest tests/test_console.py -q")
    return 0


if __name__ == "__main__":
    sys.exit(main())

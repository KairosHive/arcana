"""
What colour transfer needs, and what it should say when it does not have it.

Arcana used to gate the neural method on finding the upstream ModFlows *source*
tree. arcana/modflows_net.py replaced that source -- it is our own
implementation of the same inference path, written because the upstream repo
carries no licence (LICENSING.md, item A0) -- but the check stayed. So on a
machine with the code and the MIT checkpoint both present and colour transfer
running in about five seconds, the panel said:

    ModFlows is not installed in this build - use the LAB method.

The only thing that can actually be missing now is the checkpoint, and that is
a download rather than an absence.

Run with:  python -m pytest tests/test_color_transfer_status.py -q
"""

import os
import pathlib
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arcana import color_transfer as ct  # noqa: E402


def test_readiness_is_about_the_checkpoint_and_nothing_else(monkeypatch, tmp):
    fake = pathlib.Path(tmp) / "modflows_color_encoder_B6_dim_8195_iter_700000.pt"
    fake.write_bytes(b"not really a checkpoint")

    monkeypatch.setattr(ct, "checkpoint_path", lambda: fake)
    st = ct.status()
    assert st["ready"] is True
    assert st["checkpoint"] == str(fake)

    monkeypatch.setattr(ct, "checkpoint_path", lambda: None)
    st = ct.status()
    assert st["ready"] is False
    assert st["checkpoint"] is None
    # Still says where it would go, because the UI offers to fetch it.
    assert st["download_to"]
    assert st["download_mb"] > 0


def test_status_no_longer_reports_a_source_tree():
    """
    A leftover "source" key is how the false message survived: the card asked
    for it, found nothing, and concluded the build was broken.
    """
    assert "source" not in ct.status()


def test_the_source_search_is_gone():
    for name in ("modflows_source_dir", "_ensure_modflows_available",
                 "ENV_MODFLOWS_DIR"):
        assert not hasattr(ct, name), f"{name} still gates colour transfer"


def test_availability_follows_the_code_we_ship():
    """
    torch and torchvision are hard dependencies, so on any environment that can
    run the app at all this is True -- which is the point. It was False here
    while transfers were completing successfully.
    """
    assert ct.COLOR_TRANSFER_AVAILABLE is True
    assert ct.COLOR_TRANSFER_ERROR is None


def test_the_checkpoint_is_still_looked_for_in_several_places(monkeypatch):
    """Deleting the source search must not have taken the checkpoint search with it."""
    dirs = ct._candidate_dirs()
    assert len(dirs) >= 2
    assert any("modflows" in str(d).lower() for d in dirs)


def test_the_false_message_is_not_shown_to_anyone():
    """
    Guards the copy, not just the condition that produced it. The sentence was
    wrong in a specific way -- it told the user to give up and use LAB while
    the neural method was working -- so it is worth failing loudly if it comes
    back.

    Only string literals are scanned. A comment explaining why the message was
    removed is exactly the kind of thing a raw substring search trips over, and
    a comment cannot be shown to anybody.
    """
    import io, tokenize

    app = pathlib.Path(ct.__file__).parent / "arcana.py"
    with open(app, "rb") as fh:
        literals = [tok.string for tok in tokenize.tokenize(fh.readline)
                    if tok.type == tokenize.STRING]
    offenders = [t for t in literals if "not installed in this build" in t]
    assert not offenders, offenders


def main():
    import tempfile
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        if "monkeypatch" in fn.__code__.co_varnames[:fn.__code__.co_argcount]:
            print(f"  skip {name} (needs pytest)")
            continue
        try:
            fn()
            print(f"  ok   {name}")
        except Exception as e:
            failures += 1
            print(f"  FAIL {name}: {type(e).__name__}: {e}")
    print("all passed" if not failures else f"{failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

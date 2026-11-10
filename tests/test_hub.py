"""
Loading models from disk first, and from the hub with retries. See arcana/hub.py.

Run with:  python -m pytest tests/test_hub.py -q
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arcana import hub  # noqa: E402


class RemoteProtocolError(Exception):
    """Named like httpx's, which is what huggingface-hub 1.x raises on a dropped connection."""


class Loader:
    def __init__(self, cached=True, drops=0, broken=None):
        self.cached, self.drops, self.broken = cached, drops, broken
        self.calls = []

    def __call__(self, name, local_files_only=False, **kw):
        self.calls.append("disk" if local_files_only else "hub")
        if local_files_only:
            if not self.cached:
                raise OSError("not in the cache")
            return f"{name} from disk"
        if self.broken:
            raise self.broken
        if self.drops:
            self.drops -= 1
            raise RemoteProtocolError("Server disconnected without sending a response.")
        return f"{name} from the hub"


def test_a_cached_model_never_asks_the_network():
    load = Loader(cached=True, drops=99)
    assert hub.pretrained(load, "clip") == "clip from disk"
    assert load.calls == ["disk"]


def test_a_missing_model_is_fetched_and_a_dropped_connection_is_tried_again(monkeypatch):
    waits = []
    monkeypatch.setattr(hub.time, "sleep", waits.append)
    load = Loader(cached=False, drops=2)
    assert hub.pretrained(load, "clip", torch_dtype="x") == "clip from the hub"
    assert load.calls == ["disk", "hub", "hub", "hub"]
    assert waits == [1.0, 3.0]


def test_it_gives_up_after_the_last_try(monkeypatch):
    monkeypatch.setattr(hub.time, "sleep", lambda s: None)
    load = Loader(cached=False, drops=99)
    with pytest.raises(RemoteProtocolError):
        hub.pretrained(load, "clip")
    assert load.calls.count("hub") == hub.RETRIES


def test_an_error_that_is_not_the_network_is_raised_at_once(monkeypatch):
    monkeypatch.setattr(hub.time, "sleep", lambda s: pytest.fail("should not wait"))
    load = Loader(cached=False, broken=ValueError("no such model"))
    with pytest.raises(ValueError, match="no such model"):
        hub.pretrained(load, "clip")
    assert load.calls == ["disk", "hub"]


def test_a_caller_that_chose_for_itself_is_left_alone():
    load = Loader(cached=False)
    with pytest.raises(OSError):
        hub.pretrained(load, "clip", local_files_only=True)
    assert load.calls == ["disk"]


def test_what_counts_as_the_network_dropping():
    assert hub.transient(RemoteProtocolError("x"))
    assert hub.transient(ConnectionResetError(10054, "forcibly closed"))
    assert hub.transient(OSError("An existing connection was forcibly closed by the remote host"))
    wrapped = RuntimeError("could not load")
    wrapped.__cause__ = RemoteProtocolError("Server disconnected")
    assert hub.transient(wrapped)
    assert not hub.transient(ValueError("bad argument"))
    assert not hub.transient(FileNotFoundError("missing"))


def test_every_loader_in_the_app_goes_through_it():
    """A new from_pretrained added without the wrapper would bring the failure back."""
    import re
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "arcana")
    bare = []
    for folder, _, names in os.walk(root):
        for name in names:
            if not name.endswith(".py") or name in ("hub.py", "omni_worker.py"):
                continue
            path = os.path.join(folder, name)
            for n, line in enumerate(open(path, encoding="utf-8"), 1):
                code = line.split("#")[0]
                if re.search(r"\.from_pretrained\(", code) or re.search(r"(?<![\w.,] )(?<![\w.])hf_hub_download\(", code):
                    bare.append(f"{os.path.relpath(path, root)}:{n}")
    assert not bare, bare

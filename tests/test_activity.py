"""
The activity feed: the ops that ran, pushed to the page as they happen.
See VIDEO_ROADMAP.md, 3.7.

Run with:  python -m pytest tests/test_activity.py -q
"""

import json
import os
import sys
import threading
import time

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from arcana import activity, ops  # noqa: E402

import synth  # noqa: E402


@pytest.fixture
def ds(tmp, monkeypatch):
    activity.clear()
    synth.use(tmp, monkeypatch)
    yield
    activity.clear()


def test_an_op_an_agent_runs_is_recorded_with_what_came_of_it(ds):
    ops.call("search text", {"dataset": "synth", "phrase": ["water"], "k": 3}, actor="agent")
    got = activity.since(0)
    assert len(got) == 1
    a = got[0]
    assert a["actor"] == "agent" and a["op"] == "search text" and a["seq"] == 1
    assert a["args"] == {"dataset": "synth", "phrase": ["water"], "k": 3}       # only what was asked, no defaults
    assert a["summary"] and time.time() - a["t"] < 5


def test_lookups_about_arcana_itself_and_refusals_leave_no_line(ds):
    ops.call("orient", {}, actor="agent")
    ops.call("op list", {}, actor="agent")
    with pytest.raises(ops.Refusal):
        ops.call("search text", {"dataset": "nope", "phrase": ["x"]}, actor="agent")
    assert activity.since(0) == [] and activity.latest() == 0


def test_writes_are_recorded_and_the_list_can_be_read_from_a_point(ds):
    ops.call("board create", {"name": "Rain"}, actor="agent")
    ops.call("board add", {"board": "rain", "items": ["synth:1", "synth:2"]}, actor="agent")
    ops.call("search text", {"dataset": "synth", "phrase": ["forest"]}, actor="cli")
    assert [a["op"] for a in activity.since(0)] == ["board create", "board add", "search text"]
    assert [a["op"] for a in activity.since(2)] == ["search text"]
    assert [a["op"] for a in activity.since(0, actor="agent")] == ["board create", "board add"]
    made = activity.since(0)[0]
    assert made["result"].get("message") or made["summary"]


def test_a_waiter_wakes_when_an_op_runs(ds):
    woke = {}

    def waiter():
        t = time.time()
        woke["seq"] = activity.wait(0, timeout=10)
        woke["took"] = time.time() - t

    th = threading.Thread(target=waiter)
    th.start()
    time.sleep(0.2)
    ops.call("search text", {"dataset": "synth", "phrase": ["water"]}, actor="agent")
    th.join(5)
    assert woke["seq"] == 1 and woke["took"] < 3
    assert activity.wait(1, timeout=0.1) == 1                 # nothing newer: it comes back at the timeout


def test_the_page_reads_the_feed_and_the_stream(ds, tmp):
    from flask import Flask
    from arcana import api
    app = Flask("t")
    api.register(app)
    c = app.test_client()
    c.environ_base["HTTP_X_ARCANA_UI"] = api.UI_TOKEN
    ops.call("search text", {"dataset": "synth", "phrase": ["water"]}, actor="agent")
    got = c.get("/api/activity").get_json()
    assert got["latest"] == 1 and got["items"][0]["op"] == "search text"
    assert c.get("/api/activity?after=1").get_json()["items"] == []

    # The stream takes the token in its address, since an event stream carries no header.
    bare = app.test_client()
    assert bare.get("/api/events").status_code == 403
    assert bare.get("/api/activity?ui=" + api.UI_TOKEN).status_code == 403     # only the stream may
    with bare.get("/api/events?ui=" + api.UI_TOKEN, buffered=False) as stream:
        assert stream.status_code == 200 and stream.mimetype == "text/event-stream"
        first = next(stream.response).decode()
        assert first.startswith("data: ")
        said = json.loads(first[6:])
        assert said["activity"] == 1 and said["signature"]
